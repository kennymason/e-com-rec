"""Train the two-tower retriever on one window, track it in MLflow, export embeddings."""

import argparse
import time
from dataclasses import asdict, fields

import mlflow
import polars as pl

from ecomrec.config import RETRIEVAL_K, ROOT, artifacts_dir, dataset_dir
from ecomrec.eval.evaluate import evaluate
from ecomrec.eval.report import add_window_args, load_window, results_header, results_row
from ecomrec.retrieval.two_tower import TwoTowerConfig, TwoTowerRetriever


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    for f in fields(TwoTowerConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            kind = int if f.default is None else type(f.default)
            parser.add_argument(flag, type=kind, default=f.default)


def _metrics(scores: dict) -> dict[str, float]:
    """Scalar scores under MLflow-safe names."""
    return {k.replace("@", "_at_"): v for k, v in scores.items() if isinstance(v, (int, float))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_window_args(parser)
    _add_config_args(parser)
    args = parser.parse_args()
    config = TwoTowerConfig(**{f.name: getattr(args, f.name) for f in fields(TwoTowerConfig)})

    split, n_catalog = load_window(args.dataset, args.window)
    data = dataset_dir(args.dataset)
    retriever = TwoTowerRetriever(
        pl.read_parquet(data / "articles.parquet"),
        pl.read_parquet(data / "customers.parquet"),
        config,
    )
    customers = split.targets["customer_idx"]
    recall = f"recall@{RETRIEVAL_K}"

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("two-tower")
    with mlflow.start_run(run_name=f"{args.dataset}-{args.window}"):
        mlflow.log_params(
            asdict(config)
            | {"dataset": args.dataset, "window": args.window, "cutoff": str(split.cutoff)}
        )
        started = time.perf_counter()

        def on_epoch(epoch: int, loss: float) -> None:
            recs = retriever.recommend(customers, RETRIEVAL_K)
            scores = evaluate(recs, split, n_catalog, n_boot=1)
            mlflow.log_metrics({"loss": loss} | _metrics(scores), step=epoch)
            print(
                f"epoch {epoch}: loss {loss:.4f}  {recall} {scores[recall]:.4f}  "
                f"ndcg@12 {scores['ndcg@12']:.4f}  ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

        retriever.fit(split.history, split.cutoff, on_epoch)

        scores = evaluate(retriever.recommend(customers, RETRIEVAL_K), split, n_catalog)
        print("\n" + results_header())
        print(results_row("Two-tower only", scores))

        out_dir = artifacts_dir("two_tower", args.dataset, args.window)
        retriever.export(out_dir, customers)
        mlflow.log_artifacts(str(out_dir))
        print(f"\nExported embeddings and weights to {out_dir}")


if __name__ == "__main__":
    main()
