from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from ecomrec.baselines import ALS, CoPurchase

CUTOFF = date(2020, 9, 15)
GROUP_A = [100, 101, 102, 103]
GROUP_B = [200, 201, 202, 203]


@pytest.fixture
def clustered() -> pl.DataFrame:
    """40 customers in two taste groups. Each buys 3 of the 4 articles of their group.
    Customers 0 (group A) and 1 (group B) are each missing the last article of their group."""
    rng = np.random.default_rng(0)
    rows = []
    for customer in range(40):
        group = GROUP_A if customer % 2 == 0 else GROUP_B
        bought = group[:3] if customer < 2 else rng.choice(group, 3, replace=False).tolist()
        for article in bought:
            day = CUTOFF - timedelta(days=int(rng.integers(1, 30)))
            rows.append((day, customer, article))
    # Leakage bait: after the cutoff, customer 0 buys everything in group B.
    rows += [(CUTOFF, 0, article) for article in GROUP_B]
    return pl.DataFrame(
        rows,
        schema={"t_dat": pl.Date, "customer_idx": pl.UInt32, "article_id": pl.Int32},
        orient="row",
    )


@pytest.mark.parametrize(
    "model", [CoPurchase(history_days=None), ALS(factors=8, iterations=30, history_days=None)]
)
def test_recommends_within_the_customers_taste_group(clustered, model):
    model.fit(clustered, CUTOFF)
    recs = model.recommend(pl.Series([0, 1], dtype=pl.UInt32), n=4)
    assert recs.schema["article_id"] == pl.List(pl.Int32)
    a, b = recs["article_id"].to_list()
    assert set(a) == set(GROUP_A)  # includes 103, which customer 0 never bought
    assert set(b) == set(GROUP_B)


@pytest.mark.parametrize(
    "model", [CoPurchase(history_days=None), ALS(factors=8, iterations=30, history_days=None)]
)
def test_unknown_customers_get_popular_items_and_lists_are_distinct(clustered, model):
    model.fit(clustered, CUTOFF)
    recs = model.recommend(pl.Series([999, 0], dtype=pl.UInt32), n=8)
    cold, warm = recs["article_id"].to_list()
    assert cold == model.popularity.ranked[:8]
    assert len(warm) == len(set(warm)) > 4  # padded past the 4 in-group articles
    assert set(warm[:4]) == set(GROUP_A)


def test_history_days_limits_the_training_window(clustered):
    model = CoPurchase(history_days=10).fit(clustered, CUTOFF)
    recent = clustered.filter(
        (pl.col("t_dat") < CUTOFF) & (pl.col("t_dat") >= CUTOFF - timedelta(days=10))
    )
    assert model.user_items.sum() == recent.height
