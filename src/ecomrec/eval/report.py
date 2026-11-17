"""Shared pieces of the evaluation CLIs: loading a window and printing results rows."""

import argparse

import polars as pl

from ecomrec.config import RETRIEVAL_K, TOP_K, dataset_dir
from ecomrec.data.split import Split, make_split, window_cutoffs

_R, _N, _M = f"recall@{RETRIEVAL_K}", f"ndcg@{TOP_K}", f"map@{TOP_K}"
_NEW, _COV, _HEAD = f"new_item_{_R}", f"coverage@{TOP_K}", f"top1pct_share@{TOP_K}"


def add_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--window", choices=["val", "test"], default="val")


def load_window(dataset: str, window: str) -> tuple[Split, int]:
    """The split for a standard window, and the catalog size."""
    data = dataset_dir(dataset)
    transactions = pl.read_parquet(data / "transactions.parquet")
    n_catalog = pl.scan_parquet(data / "articles.parquet").select(pl.len()).collect().item()
    split = make_split(transactions, window_cutoffs(transactions["t_dat"].max())[window])
    print(
        f"{dataset}/{window}: cutoff {split.cutoff}, {split.targets.height:,} customers, "
        f"{len(split.new_items):,} new items\n"
    )
    return split, n_catalog


def results_header() -> str:
    return (
        f"| Model | {_R} | {_N} (95% CI) | {_M} | {_NEW} | {_COV} | {_HEAD} |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )


def results_row(name: str, s: dict) -> str:
    lo, hi = s[f"{_N}_ci"]
    return (
        f"| {name} | {s[_R]:.4f} | {s[_N]:.4f} ({lo:.4f}–{hi:.4f}) | {s[_M]:.4f} "
        f"| {s[_NEW]:.4f} | {s[_COV]:.4f} | {s[_HEAD]:.4f} |"
    )
