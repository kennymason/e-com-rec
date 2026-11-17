from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from ecomrec.retrieval.features import (
    ITEM_ATTRIBUTES,
    age_buckets,
    build_interactions,
    build_item_table,
)
from ecomrec.retrieval.model import in_batch_softmax_loss
from ecomrec.retrieval.two_tower import TwoTowerConfig, TwoTowerRetriever

CUTOFF = date(2020, 9, 15)
CPU = torch.device("cpu")


def make_articles(article_ids: list[int]) -> pl.DataFrame:
    """Catalog where the only informative attribute is the article's hundred (its group)."""
    columns = {"article_id": pl.Series(article_ids, dtype=pl.Int32)}
    for name in ITEM_ATTRIBUTES:
        columns[name] = pl.Series(["x"] * len(article_ids))
    return pl.DataFrame(columns).with_columns((pl.col("article_id") // 100).alias("department_no"))


def make_history(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """Rows are (days before the cutoff, customer, article). All bought in store."""
    return pl.DataFrame(
        {
            "t_dat": [CUTOFF - timedelta(days=d) for d, _, _ in rows],
            "customer_idx": [c for _, c, _ in rows],
            "article_id": [a for _, _, a in rows],
            "sales_channel_id": [1] * len(rows),
        },
        schema={
            "t_dat": pl.Date,
            "customer_idx": pl.UInt32,
            "article_id": pl.Int32,
            "sales_channel_id": pl.Int8,
        },
    )


def test_item_table_hides_the_id_of_unseen_articles():
    table = build_item_table(make_articles([300, 100, 200]), seen_article_ids=np.array([100, 300]))
    assert table.article_ids.tolist() == [100, 200, 300]
    assert table.rows(np.array([300, 100])).tolist() == [3, 1]
    assert table.features[:, 0].tolist() == [0, 1, 0, 3]  # padding row, then 200 is unseen
    assert (table.features[0] == 0).all()
    assert (table.features[1:, 1:] >= 1).all()
    department = 1 + ITEM_ATTRIBUTES.index("department_no")
    assert table.features[1:, department].tolist() == [1, 2, 3]
    assert table.cardinalities[0] == 4 and table.cardinalities[department] == 4


def test_age_buckets():
    customers = pl.DataFrame(
        {"customer_idx": [0, 2, 3], "age": [19, None, 44]},
        schema={"customer_idx": pl.UInt32, "age": pl.Int8},
    )
    assert age_buckets(customers).tolist() == [1, 0, 0, 6]  # idx 1 is absent, idx 2 has no age


def test_interaction_pointers_separate_earlier_days_from_the_same_day():
    # customer 5: one purchase 9 days ago, two purchases 3 days ago. customer 7: one purchase.
    history = make_history([(3, 5, 101), (9, 5, 100), (3, 5, 102), (4, 7, 200)])
    items = build_item_table(make_articles([100, 101, 102, 200]), np.array([100, 101, 102, 200]))
    inter = build_interactions(history, items, n_customers=9)

    assert inter.customer.tolist() == [5, 5, 5, 7]
    assert inter.item_row[0] == 1  # the oldest purchase sorts first
    assert inter.cust_start.tolist() == [0, 0, 0, 3]
    # Both same-day purchases see only the earlier one as history, never each other.
    assert inter.day_start.tolist() == [0, 1, 1, 3]
    assert (inter.first_row[5], inter.end_row[5]) == (0, 3)
    assert (inter.first_row[7], inter.end_row[7]) == (3, 4)
    assert (inter.first_row[0], inter.end_row[0]) == (0, 0)  # no history


def test_loss_masks_duplicate_items_and_applies_logq():
    users = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    items = users.clone()
    same_item = torch.zeros(4, dtype=torch.long)
    # Every other row holds the same item, so each row has no negatives left.
    assert in_batch_softmax_loss(users, items, same_item, None, 0.1).item() == pytest.approx(0)

    rows = torch.arange(4)
    plain = in_batch_softmax_loss(users, items, rows, None, 0.1)
    uniform = in_batch_softmax_loss(users, items, rows, torch.full((4,), -2.0), 0.1)
    skewed = in_batch_softmax_loss(users, items, rows, torch.tensor([0.0, -5, -5, -5]), 0.1)
    assert uniform.item() == pytest.approx(plain.item(), abs=1e-5)  # a constant shift cancels
    assert skewed.item() != pytest.approx(plain.item(), abs=1e-3)


@pytest.fixture(scope="module")
def fitted() -> TwoTowerRetriever:
    """60 customers in two taste groups (articles 100-109 and 200-209), 6 purchases each on
    distinct days. Customers 0 and 1 have only bought the first 6 articles of their group."""
    rng = np.random.default_rng(0)
    article_ids = list(range(100, 110)) + list(range(200, 210))
    rows = []
    for customer in range(60):
        base = 100 if customer % 2 == 0 else 200
        picks = range(6) if customer < 2 else rng.choice(10, 6, replace=False)
        days = rng.choice(np.arange(1, 40), 6, replace=False)
        rows += [(int(d), customer, base + int(p)) for d, p in zip(days, picks, strict=True)]
    customers = pl.DataFrame(
        {"customer_idx": range(60), "age": [30] * 60},
        schema={"customer_idx": pl.UInt32, "age": pl.Int8},
    )
    config = TwoTowerConfig(dim=16, hidden=32, history_len=4, epochs=40, batch_size=64, lr=3e-3)
    retriever = TwoTowerRetriever(make_articles(article_ids), customers, config, CPU)
    return retriever.fit(make_history(rows), CUTOFF)


def test_training_history_only_contains_earlier_days(fitted):
    # Rebuild what fit() saw: for every row, its history window must end before its own day.
    item_row = fitted.item_row.numpy()
    first, end = fitted.first_row.numpy(), fitted.end_row.numpy()
    for customer in range(60):
        ends = torch.arange(first[customer], end[customer])
        starts = torch.full_like(ends, first[customer])
        hist, _, gap = fitted._history(ends, starts, fitted.cutoff_day)
        assert ((gap >= 1) | (ends == first[customer])).all()
        for position, window in zip(ends.tolist(), hist.tolist(), strict=True):
            expected = item_row[max(first[customer], position - 4) : position][::-1].tolist()
            assert [r for r in window if r > 0] == expected


def test_recommends_within_the_customers_taste_group(fitted):
    recs = fitted.recommend(pl.Series(range(60), dtype=pl.UInt32), n=5)
    assert recs.schema["article_id"] == pl.List(pl.Int32)
    in_group = [
        np.mean([a // 100 == (1 if c % 2 == 0 else 2) for a in articles])
        for c, articles in zip(recs["customer_idx"], recs["article_id"], strict=True)
    ]
    assert np.mean(in_group) > 0.95
    assert all(len(set(a)) == 5 for a in recs["article_id"].to_list())


def test_export_matches_recommend(fitted, tmp_path):
    customers = pl.Series([0, 1, 2], dtype=pl.UInt32)
    fitted.export(tmp_path, customers)
    users = np.load(tmp_path / "user_vectors.npy")
    items = np.load(tmp_path / "item_vectors.npy")
    ids = np.load(tmp_path / "item_article_ids.npy")
    assert np.allclose(np.linalg.norm(users, axis=1), 1, atol=1e-5)
    top = ids[np.argsort(-(users @ items.T), axis=1)[:, :3]]
    assert top.tolist() == fitted.recommend(customers, n=3)["article_id"].to_list()
    assert (tmp_path / "model.pt").exists()
