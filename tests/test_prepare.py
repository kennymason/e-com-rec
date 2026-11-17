from datetime import date

import polars as pl

from ecomrec.data.prepare import prepare

CUSTOMERS = """customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code
aaa,,,ACTIVE,NONE,49.0,p1
bbb,1.0,1.0,ACTIVE,Regularly,,p2
ccc,,,PRE-CREATE,NONE,21.0,p3
"""
ARTICLES = """article_id,product_code,prod_name
0108775015,0108775,Strap top
0706016001,0706016,Jade denim
"""
TRANSACTIONS = """t_dat,customer_id,article_id,price,sales_channel_id
2020-09-02,bbb,0706016001,0.0339,2
2020-09-01,aaa,0108775015,0.0085,1
2020-09-01,bbb,0108775015,0.0085,2
"""


def write_raw(raw):
    (raw / "customers.csv").write_text(CUSTOMERS)
    (raw / "articles.csv").write_text(ARTICLES)
    (raw / "transactions_train.csv").write_text(TRANSACTIONS)


def test_prepare_writes_typed_sorted_tables(tmp_path):
    write_raw(tmp_path)
    out = tmp_path / "out"
    assert prepare(tmp_path, out) == {"customers": 3, "articles": 2, "transactions": 3}

    customers = pl.read_parquet(out / "customers.parquet")
    assert customers["customer_idx"].to_list() == [0, 1, 2]
    assert customers["age"].to_list() == [49, None, 21]
    assert customers["FN"].to_list() == [0, 1, 0]
    assert "postal_code" not in customers.columns

    assert pl.read_parquet(out / "articles.parquet")["article_id"].to_list() == [
        108775015,
        706016001,
    ]

    tx = pl.read_parquet(out / "transactions.parquet")
    assert tx.schema == {
        "t_dat": pl.Date,
        "customer_idx": pl.UInt32,
        "article_id": pl.Int32,
        "price": pl.Float32,
        "sales_channel_id": pl.Int8,
    }
    assert tx.select("t_dat", "customer_idx", "article_id").rows() == [
        (date(2020, 9, 1), 0, 108775015),
        (date(2020, 9, 1), 1, 108775015),
        (date(2020, 9, 2), 1, 706016001),
    ]


def test_sampling_keeps_customer_idx_stable_and_filters_transactions(tmp_path):
    write_raw(tmp_path)
    prepare(tmp_path, tmp_path / "full")
    prepare(tmp_path, tmp_path / "sample", sample_frac=0.5)

    full = pl.read_parquet(tmp_path / "full" / "customers.parquet")
    sample = pl.read_parquet(tmp_path / "sample" / "customers.parquet")
    assert sample.height <= full.height
    assert sample.join(full, on=["customer_idx", "customer_id"]).height == sample.height

    tx = pl.read_parquet(tmp_path / "sample" / "transactions.parquet")
    assert set(tx["customer_idx"]) <= set(sample["customer_idx"])
