"""Evaluate the baseline ladder on a time-based window and print a results table."""

import argparse

from ecomrec.baselines import ALS, CoPurchase, Popularity, Repurchase
from ecomrec.config import RETRIEVAL_K
from ecomrec.eval.evaluate import evaluate
from ecomrec.eval.report import add_window_args, load_window, results_header, results_row

MODELS = {
    "Popularity": Popularity,
    "Repurchase + popular": Repurchase,
    "Item-to-item co-purchase": CoPurchase,
    "ALS": ALS,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_window_args(parser)
    args = parser.parse_args()

    split, n_catalog = load_window(args.dataset, args.window)
    customers = split.targets["customer_idx"]
    print(results_header())
    for name, cls in MODELS.items():
        recs = cls().fit(split.history, split.cutoff).recommend(customers, RETRIEVAL_K)
        print(results_row(name, evaluate(recs, split, n_catalog)))


if __name__ == "__main__":
    main()
