"""Baseline ladder. Every model exposes the same two methods:

- ``fit(history, cutoff)``: learn from transactions strictly before ``cutoff``
- ``recommend(customers, n)``: return ``customer_idx``, ``article_id`` (ranked list of n)
"""

from ecomrec.baselines.implicit_models import ALS, CoPurchase
from ecomrec.baselines.popularity import Popularity
from ecomrec.baselines.repurchase import Repurchase

__all__ = ["ALS", "CoPurchase", "Popularity", "Repurchase"]
