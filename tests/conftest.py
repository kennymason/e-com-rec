from datetime import date

import polars as pl
import pytest

# (day of September 2020, customer, article). Test window starts on the 15th.
ROWS = [
    (1, 1, 10),
    (2, 1, 11),
    (3, 2, 10),
    (9, 1, 10),
    (10, 2, 12),
    (10, 3, 10),
    (12, 3, 12),
    (14, 2, 12),
    # --- test week ---
    (15, 1, 11),
    (16, 2, 13),  # article 13 is new
    (21, 3, 12),
    (21, 3, 12),  # duplicate purchase
    (21, 4, 10),  # customer 4 has no history
]


@pytest.fixture
def transactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_dat": [date(2020, 9, d) for d, _, _ in ROWS],
            "customer_idx": [c for _, c, _ in ROWS],
            "article_id": [a for _, _, a in ROWS],
        },
        schema={"t_dat": pl.Date, "customer_idx": pl.UInt32, "article_id": pl.Int32},
    )
