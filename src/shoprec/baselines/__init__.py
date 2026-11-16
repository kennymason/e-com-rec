"""Baseline ladder. Every model exposes the same two methods:

- ``fit(history, cutoff)``: learn from transactions strictly before ``cutoff``
- ``recommend(customers, n)``: return ``customer_idx``, ``article_id`` (ranked list of n)
"""

from shoprec.baselines.popularity import Popularity
from shoprec.baselines.repurchase import Repurchase

__all__ = ["Popularity", "Repurchase"]
