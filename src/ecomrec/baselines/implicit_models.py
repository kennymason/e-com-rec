"""Rungs 3 and 4: item-to-item co-purchase and ALS, both via the ``implicit`` library.

Both models work on a customer x article matrix of purchase counts built from history.
Items the customer already bought are not filtered out, because repurchase is real signal
in fashion. Customers with no usable history fall back to popularity, and short lists are
padded with popular items.

Both default to the last 28 days of history. On the full-data validation week that beat
longer windows clearly (56 days and up) and was level with 14 days.
"""

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix

from ecomrec.baselines.popularity import Popularity

# implicit converts its own internal matrix and warns about it on every fit
warnings.filterwarnings("ignore", message="Method expects CSR input")


class _ImplicitBaseline:
    #: drop recommendations scoring at or below this (None keeps everything)
    min_score: float | None = None

    def __init__(self, history_days: int | None = None, popularity_days: int = 7) -> None:
        self.history_days = history_days
        self.popularity = Popularity(popularity_days)

    def _make_model(self):
        raise NotImplementedError

    def fit(self, history: pl.DataFrame, cutoff: date):
        self.popularity.fit(history, cutoff)
        history = history.filter(pl.col("t_dat") < cutoff)
        if self.history_days is not None:
            start = cutoff - timedelta(days=self.history_days)
            history = history.filter(pl.col("t_dat") >= start)
        counts = history.group_by("customer_idx", "article_id").len()

        self.users = np.unique(counts["customer_idx"].to_numpy())
        self.items = np.unique(counts["article_id"].to_numpy())
        rows = np.searchsorted(self.users, counts["customer_idx"].to_numpy())
        cols = np.searchsorted(self.items, counts["article_id"].to_numpy())
        self.user_items = csr_matrix(
            (counts["len"].to_numpy().astype(np.float32), (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )
        self.model = self._make_model()
        self.model.fit(self.user_items, show_progress=False)
        return self

    def recommend(self, customers: pl.Series, n: int) -> pl.DataFrame:
        popular = self.popularity.ranked[:n]
        wanted = customers.to_numpy()
        pos = np.minimum(np.searchsorted(self.users, wanted), len(self.users) - 1)
        known = self.users[pos] == wanted

        lists: list[list[int]] = [popular] * len(wanted)
        if known.any():
            ids, scores = self.model.recommend(
                pos[known],
                self.user_items[pos[known]],
                N=min(n, len(self.items)),
                filter_already_liked_items=False,
            )
            valid = ids >= 0
            if self.min_score is not None:
                valid &= scores > self.min_score
            for row, i in enumerate(np.flatnonzero(known)):
                own = self.items[ids[row][valid[row]]].tolist()
                if len(own) < n:
                    taken = set(own)
                    own = (own + [a for a in popular if a not in taken])[:n]
                lists[i] = own
        return pl.DataFrame(
            {"customer_idx": customers, "article_id": lists},
            schema_overrides={"article_id": pl.List(pl.Int32)},
        )


class CoPurchase(_ImplicitBaseline):
    """Item-to-item "bought together": cosine similarity between article purchase columns,
    summed over the articles the customer has bought."""

    min_score = 0.0

    def __init__(self, neighbours: int = 50, history_days: int | None = 28) -> None:
        super().__init__(history_days)
        self.neighbours = neighbours

    def _make_model(self):
        return CosineRecommender(K=self.neighbours)


class ALS(_ImplicitBaseline):
    """Implicit-feedback matrix factorization (Hu, Koren, Volinsky) trained with ALS."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        history_days: int | None = 28,
        seed: int = 0,
    ) -> None:
        super().__init__(history_days)
        self.params = dict(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            random_state=seed,
        )

    def _make_model(self):
        return AlternatingLeastSquares(**self.params)
