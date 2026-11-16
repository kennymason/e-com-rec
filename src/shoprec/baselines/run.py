"""Evaluate the baseline ladder on a time-based window and print a results table."""

import argparse

import polars as pl

from shoprec.baselines import Popularity, Repurchase
from shoprec.config import RETRIEVAL_K, TOP_K, dataset_dir
from shoprec.data.split import make_split, window_cutoffs
from shoprec.eval.evaluate import evaluate

MODELS = {
    "Popularity": Popularity,
    "Repurchase + popular": Repurchase,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="full")
    parser.add_argument("--window", choices=["val", "test"], default="val")
    args = parser.parse_args()

    data = dataset_dir(args.dataset)
    transactions = pl.read_parquet(data / "transactions.parquet")
    n_catalog = pl.scan_parquet(data / "articles.parquet").select(pl.len()).collect().item()

    cutoff = window_cutoffs(transactions["t_dat"].max())[args.window]
    split = make_split(transactions, cutoff)
    customers = split.targets["customer_idx"]
    print(
        f"{args.dataset}/{args.window}: cutoff {cutoff}, {len(customers):,} customers, "
        f"{len(split.new_items):,} new items\n"
    )

    r, n, m = f"recall@{RETRIEVAL_K}", f"ndcg@{TOP_K}", f"map@{TOP_K}"
    new, cov, head = f"new_item_{r}", f"coverage@{TOP_K}", f"top1pct_share@{TOP_K}"
    print(f"| Model | {r} | {n} (95% CI) | {m} | {new} | {cov} | {head} |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, cls in MODELS.items():
        recs = cls().fit(split.history, cutoff).recommend(customers, RETRIEVAL_K)
        s = evaluate(recs, split, n_catalog)
        lo, hi = s[f"{n}_ci"]
        print(
            f"| {name} | {s[r]:.4f} | {s[n]:.4f} ({lo:.4f}–{hi:.4f}) | {s[m]:.4f} "
            f"| {s[new]:.4f} | {s[cov]:.4f} | {s[head]:.4f} |"
        )


if __name__ == "__main__":
    main()
