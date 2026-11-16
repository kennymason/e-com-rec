from datetime import date

import polars as pl
import pytest

from shoprec.data.split import make_split, window_cutoffs


def test_window_cutoffs_are_final_two_weeks():
    cutoffs = window_cutoffs(date(2020, 9, 22))
    assert cutoffs == {"val": date(2020, 9, 9), "test": date(2020, 9, 16)}


def test_history_never_reaches_the_cutoff(transactions):
    cutoff = date(2020, 9, 15)
    split = make_split(transactions, cutoff)
    assert split.history["t_dat"].max() < cutoff
    assert split.history.height == 8


def test_targets_cover_only_the_window(transactions):
    split = make_split(transactions, date(2020, 9, 8))  # val window: 8th to 14th
    assert split.targets.to_dict(as_series=False) == {
        "customer_idx": [1, 2, 3],
        "article_id": [[10], [12], [10, 12]],
    }


def test_targets_are_distinct_articles_per_customer(transactions):
    split = make_split(transactions, date(2020, 9, 15))
    assert split.targets.to_dict(as_series=False) == {
        "customer_idx": [1, 2, 3, 4],
        "article_id": [[11], [13], [12], [10]],
    }


def test_new_items_have_no_sale_in_history(transactions):
    split = make_split(transactions, date(2020, 9, 15))
    assert split.new_items.to_list() == [13]
    # article 12 first sells on the 10th, so it is new for the val window
    assert make_split(transactions, date(2020, 9, 8)).new_items.to_list() == [12]


def test_empty_window_raises(transactions):
    with pytest.raises(ValueError, match="no transactions"):
        make_split(transactions, date(2021, 1, 1))


def test_history_and_window_partition_the_data(transactions):
    split = make_split(transactions, date(2020, 9, 15))
    in_window = transactions.filter(pl.col("t_dat") >= date(2020, 9, 15))
    assert split.history.height + in_window.height == transactions.height
