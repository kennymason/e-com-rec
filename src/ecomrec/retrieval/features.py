"""Feature construction for the two-tower model (numpy and polars only).

Items are addressed by *row*: the position of the article in the sorted catalog, plus one.
Row 0 is padding. Every categorical code also reserves 0 for "unknown / padding".
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

ITEM_ATTRIBUTES = [
    "product_code",
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
]

AGE_EDGES = [20, 25, 30, 35, 40, 50, 60, 70]
N_AGE_BUCKETS = len(AGE_EDGES) + 2  # 0 = unknown, then one per interval
N_CHANNEL_BUCKETS = 4  # 0 = no history, 1 = mostly store, 2 = mixed, 3 = mostly online
GAP_EDGES = [8, 15, 31, 91, 181, 366]  # days since the customer's previous purchase
N_GAP_BUCKETS = len(GAP_EDGES) + 2  # 0 = no history, then one per interval


@dataclass(frozen=True)
class ItemTable:
    article_ids: np.ndarray
    """Sorted catalog article ids. ``article_ids[row - 1]`` is the article of a row."""
    features: np.ndarray
    """``[n_items + 1, 1 + len(ITEM_ATTRIBUTES)]`` int64 codes. Column 0 is the id code: the
    row itself for articles with a sale in history, 0 otherwise, so an unseen article is
    represented by its attributes alone. Row 0 is all zeros."""
    cardinalities: list[int]
    """Number of codes per feature column, including the reserved 0."""

    def rows(self, article_ids: np.ndarray) -> np.ndarray:
        """Rows of the given article ids. Ids must be in the catalog."""
        return np.searchsorted(self.article_ids, article_ids) + 1


def build_item_table(articles: pl.DataFrame, seen_article_ids: np.ndarray) -> ItemTable:
    articles = articles.sort("article_id")
    article_ids = articles["article_id"].to_numpy()
    n = len(article_ids)

    codes = articles.select(
        pl.col(c).rank("dense").cast(pl.Int64).alias(c) for c in ITEM_ATTRIBUTES
    ).to_numpy()
    features = np.zeros((n + 1, 1 + len(ITEM_ATTRIBUTES)), dtype=np.int64)
    features[1:, 1:] = codes
    seen = np.isin(article_ids, seen_article_ids)
    features[1:, 0] = np.where(seen, np.arange(1, n + 1), 0)
    cardinalities = [n + 1] + [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]
    return ItemTable(article_ids, features, cardinalities)


def age_buckets(customers: pl.DataFrame) -> np.ndarray:
    """Age bucket per ``customer_idx`` (dense array; 0 for unknown age or customer)."""
    idx = customers["customer_idx"].to_numpy()
    age = customers["age"].to_numpy()  # float with NaN for nulls
    buckets = np.zeros(int(idx.max()) + 1, dtype=np.int64)
    known = ~np.isnan(age)
    buckets[idx[known]] = np.digitize(age[known], AGE_EDGES) + 1
    return buckets


@dataclass(frozen=True)
class Interactions:
    """Purchase history sorted by (customer, date), with the pointers needed to read any
    customer's purchases *before a given day* without touching later rows."""

    item_row: np.ndarray
    online: np.ndarray
    """1 if the purchase was made online, else 0."""
    customer: np.ndarray
    day: np.ndarray
    """Days since the epoch."""
    cust_start: np.ndarray
    """For each row, the first row of the same customer."""
    day_start: np.ndarray
    """For each row, the first row of the same customer on the same day. Everything in
    ``[cust_start, day_start)`` happened on an earlier day."""
    first_row: np.ndarray
    end_row: np.ndarray
    """Dense per ``customer_idx``: the customer's rows are ``[first_row, end_row)``.
    Both are 0 for customers with no history."""


def build_interactions(history: pl.DataFrame, items: ItemTable, n_customers: int) -> Interactions:
    history = history.sort("customer_idx", "t_dat")
    customer = history["customer_idx"].to_numpy().astype(np.int64)
    day = history["t_dat"].cast(pl.Int32).to_numpy()
    n = len(customer)
    positions = np.arange(n)

    new_customer = np.ones(n, dtype=bool)
    new_customer[1:] = customer[1:] != customer[:-1]
    new_day = new_customer.copy()
    new_day[1:] |= day[1:] != day[:-1]
    cust_start = np.maximum.accumulate(np.where(new_customer, positions, 0))
    day_start = np.maximum.accumulate(np.where(new_day, positions, 0))

    first_row = np.zeros(n_customers, dtype=np.int64)
    end_row = np.zeros(n_customers, dtype=np.int64)
    starts = positions[new_customer]
    ids = customer[starts]
    first_row[ids] = starts
    end_row[ids] = np.append(starts[1:], n)

    return Interactions(
        item_row=items.rows(history["article_id"].to_numpy()),
        online=(history["sales_channel_id"].to_numpy() == 2).astype(np.int64),
        customer=customer,
        day=day,
        cust_start=cust_start,
        day_start=day_start,
        first_row=first_row,
        end_row=end_row,
    )
