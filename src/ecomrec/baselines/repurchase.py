"""Rung 2: the customer's own purchases, most recent first, padded with popular items."""

from datetime import date

import polars as pl

from ecomrec.baselines.popularity import Popularity


class Repurchase:
    def __init__(self, window_days: int = 7) -> None:
        self.popularity = Popularity(window_days)
        self.own = pl.DataFrame()

    def fit(self, history: pl.DataFrame, cutoff: date) -> "Repurchase":
        self.popularity.fit(history, cutoff)
        self.own = (
            history.filter(pl.col("t_dat") < cutoff)
            .group_by("customer_idx", "article_id")
            .agg(pl.col("t_dat").max().alias("last"), pl.len().alias("count"))
            .sort(
                ["customer_idx", "last", "count", "article_id"],
                descending=[False, True, True, False],
            )
            .group_by("customer_idx", maintain_order=True)
            .agg(pl.col("article_id").alias("own"))
        )
        return self

    def recommend(self, customers: pl.Series, n: int) -> pl.DataFrame:
        popular = self.popularity.ranked[:n]
        joined = pl.DataFrame({"customer_idx": customers}).join(
            self.own, on="customer_idx", how="left", maintain_order="left"
        )
        lists = []
        for own in joined["own"].to_list():
            own = (own or [])[:n]
            taken = set(own)
            lists.append((own + [a for a in popular if a not in taken])[:n])
        return pl.DataFrame(
            {"customer_idx": joined["customer_idx"], "article_id": lists},
            schema_overrides={"article_id": pl.List(pl.Int32)},
        )
