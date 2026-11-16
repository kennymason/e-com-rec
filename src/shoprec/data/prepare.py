"""Convert the raw H&M CSVs into compact Parquet tables.

Output tables (in ``data/processed/<name>/``):

- ``customers.parquet``     customer_idx (u32), customer_id (original hash), age, ...
- ``articles.parquet``      article_id (i32) plus the 24 attribute columns
- ``transactions.parquet``  t_dat (date), customer_idx, article_id, price, sales_channel_id

``customer_idx`` replaces the 64-character customer hash everywhere downstream. It is
assigned from the row order of ``customers.csv`` before sampling, so the same customer has
the same index in the full and the sampled datasets.

``article_id`` is stored as an integer, which drops the leading zero of the raw id. Image
paths need the zero-padded form: ``f"{article_id:010d}"``.
"""

import argparse
from pathlib import Path

import polars as pl

from shoprec.config import RAW_DIR, dataset_dir


def prepare(
    raw_dir: Path, out_dir: Path, sample_frac: float | None = None, seed: int = 0
) -> dict[str, int]:
    """Write the three Parquet tables and return their row counts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = (
        pl.scan_csv(
            raw_dir / "customers.csv",
            schema_overrides={"age": pl.Float32, "FN": pl.Float32, "Active": pl.Float32},
        )
        .with_row_index("customer_idx")
        .select(
            "customer_idx",
            "customer_id",
            pl.col("age").cast(pl.Int8),
            "club_member_status",
            "fashion_news_frequency",
            pl.col("FN").fill_null(0).cast(pl.Int8),
            pl.col("Active").fill_null(0).cast(pl.Int8),
        )
    )
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")
        keep = pl.col("customer_idx").hash(seed) % 1000 < round(sample_frac * 1000)
        customers = customers.filter(keep)
    customers = customers.collect()
    customers.write_parquet(out_dir / "customers.parquet")

    pl.scan_csv(raw_dir / "articles.csv", schema_overrides={"article_id": pl.Int32}).sink_parquet(
        out_dir / "articles.parquet"
    )

    (
        pl.scan_csv(
            raw_dir / "transactions_train.csv",
            schema_overrides={
                "t_dat": pl.String,
                "article_id": pl.Int32,
                "price": pl.Float32,
                "sales_channel_id": pl.Int8,
            },
        )
        .join(customers.lazy().select("customer_id", "customer_idx"), on="customer_id")
        .select(
            pl.col("t_dat").str.to_date("%Y-%m-%d"),
            "customer_idx",
            "article_id",
            "price",
            "sales_channel_id",
        )
        .sort("t_dat", "customer_idx")
        .sink_parquet(out_dir / "transactions.parquet")
    )

    return {
        name: pl.scan_parquet(out_dir / f"{name}.parquet").select(pl.len()).collect().item()
        for name in ("customers", "articles", "transactions")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--name", default="full", help="output dataset name (default: full)")
    parser.add_argument("--sample-frac", type=float, help="keep this fraction of customers")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = dataset_dir(args.name)
    counts = prepare(args.raw_dir, out_dir, args.sample_frac, args.seed)
    print(f"Wrote {out_dir}")
    for name, n in counts.items():
        print(f"  {name:<13}{n:>12,} rows")


if __name__ == "__main__":
    main()
