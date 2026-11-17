import math
from datetime import date

import polars as pl
import pytest

from ecomrec.baselines import Popularity, Repurchase
from ecomrec.data.split import make_split
from ecomrec.eval.evaluate import evaluate

CUTOFF = date(2020, 9, 15)


@pytest.fixture
def split(transactions):
    return make_split(transactions, CUTOFF)


def test_popularity_uses_only_the_trailing_window(split):
    model = Popularity(window_days=7).fit(split.history, CUTOFF)
    # 8th to 14th: article 12 sold 3 times, article 10 twice; article 11 not at all
    assert model.ranked == [12, 10]
    recs = model.recommend(split.targets["customer_idx"], n=1)
    assert recs["article_id"].to_list() == [[12]] * 4


def test_popularity_ignores_rows_at_or_after_the_cutoff(transactions):
    model = Popularity(window_days=7).fit(transactions, CUTOFF)  # unfiltered on purpose
    assert model.ranked == [12, 10]


def test_repurchase_orders_by_recency_and_pads_with_popular(split):
    model = Repurchase().fit(split.history, CUTOFF)
    recs = model.recommend(split.targets["customer_idx"], n=3)
    assert recs.to_dict(as_series=False) == {
        "customer_idx": [1, 2, 3, 4],
        "article_id": [[10, 11, 12], [12, 10], [12, 10], [12, 10]],
    }
    assert recs.schema["article_id"] == pl.List(pl.Int32)


def test_evaluate_repurchase_end_to_end(split):
    recs = Repurchase().fit(split.history, CUTOFF).recommend(split.targets["customer_idx"], n=3)
    scores = evaluate(recs, split, n_catalog=4, n_boot=50)
    rank2 = 1 / math.log2(3)
    assert scores["n_customers"] == 4
    assert scores["recall@100"] == pytest.approx(3 / 4)
    assert scores["map@12"] == pytest.approx((0.5 + 0 + 1 + 0.5) / 4)
    assert scores["ndcg@12"] == pytest.approx((rank2 + 0 + 1 + rank2) / 4)
    assert scores["n_new_item_customers"] == 1
    assert scores["new_item_recall@100"] == 0.0
    assert scores["coverage@12"] == pytest.approx(3 / 4)
    assert scores["top1pct_share@12"] == pytest.approx(4 / 9)
    lo, hi = scores["ndcg@12_ci"]
    assert lo <= scores["ndcg@12"] <= hi


def test_customers_without_recommendations_score_zero(split):
    recs = pl.DataFrame(
        {"customer_idx": [3], "article_id": [[12]]},
        schema={"customer_idx": pl.UInt32, "article_id": pl.List(pl.Int32)},
    )
    scores = evaluate(recs, split, n_catalog=4, n_boot=50)
    assert scores["recall@100"] == pytest.approx(1 / 4)
