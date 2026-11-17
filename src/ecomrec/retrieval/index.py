"""FAISS index over exported item embeddings (no torch in this module).

Run as a script to evaluate retrieval through the index, from the embeddings that
``ecomrec.retrieval.train`` exported for the same dataset and window.
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np
import polars as pl

from ecomrec.config import RETRIEVAL_K, artifacts_dir
from ecomrec.eval.evaluate import evaluate
from ecomrec.eval.report import add_window_args, load_window, results_header, results_row


class ItemIndex:
    """Exact inner-product index. Vectors are L2-normalized, so this is cosine similarity."""

    def __init__(self, vectors: np.ndarray, article_ids: np.ndarray) -> None:
        self.article_ids = np.asarray(article_ids)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k article ids and scores for each query vector."""
        k = min(k, self.index.ntotal)
        scores, positions = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return self.article_ids[positions], scores

    @classmethod
    def load(cls, export_dir: Path) -> "ItemIndex":
        return cls(
            np.load(export_dir / "item_vectors.npy"), np.load(export_dir / "item_article_ids.npy")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_window_args(parser)
    args = parser.parse_args()

    export_dir = artifacts_dir("two_tower", args.dataset, args.window)
    index = ItemIndex.load(export_dir)
    users = np.load(export_dir / "user_vectors.npy")
    customers = np.load(export_dir / "user_customer_idx.npy")

    start = time.perf_counter()
    article_ids, _ = index.search(users, RETRIEVAL_K)
    elapsed = time.perf_counter() - start
    print(
        f"FAISS flat index: {index.index.ntotal:,} items, {len(users):,} queries in "
        f"{elapsed:.2f}s ({1e6 * elapsed / len(users):.0f} µs per query, batched)\n"
    )

    recs = pl.DataFrame(
        {"customer_idx": customers, "article_id": article_ids},
        schema_overrides={"article_id": pl.List(pl.Int32)},
    )
    split, n_catalog = load_window(args.dataset, args.window)
    print(results_header())
    print(results_row("Two-tower only (FAISS)", evaluate(recs, split, n_catalog)))


if __name__ == "__main__":
    main()
