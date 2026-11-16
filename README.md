# ShopRec

Two-stage e-commerce recommender: two-tower retrieval + FAISS + LightGBM ranking on 31M H&M
transactions, with multimodal cold-start and a FastAPI service under 50 ms p99.

**Status:** Week 1 in progress. The data pipeline, time-based split, evaluation harness and the
first two baselines are in place. Nothing has been run on the real data yet.

## Results

Time-based holdout, final week as test. Filled in as each model lands.

| Model | Recall@100 | NDCG@12 | MAP@12 | New-item Recall@100 |
| --- | --- | --- | --- | --- |
| Popularity | – | – | – | – |
| Repurchase + popular | – | – | – | – |
| Item-to-item co-purchase | – | – | – | – |
| ALS | – | – | – | – |
| Two-tower only | – | – | – | – |
| Two-tower + ranker | – | – | – | – |
| + content item tower | – | – | – | – |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account that has accepted the
[competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).

```bash
cp .env.example .env    # then add your Kaggle API token
make setup              # install dependencies (Python 3.12 is fetched by uv)
make download           # the three CSV tables, about 3.5 GB; images are skipped for now
make prepare-sample     # CSV -> Parquet for 10% of customers
make baselines DATASET=sample WINDOW=val
make test
```

`make prepare` and `make baselines` do the same on the full data.

## Layout

```
src/shoprec/
  config.py           paths and constants (k=12, 7-day windows)
  data/prepare.py     raw CSV -> typed Parquet, optional customer sampling
  data/split.py       cutoff-date splits, val/test windows, new-items slice
  eval/metrics.py     Recall@k, MAP@k (Kaggle definition), NDCG@k, bootstrap CI
  eval/evaluate.py    scores a model's recommendations on a split
  baselines/          popularity, repurchase + popular, and the runner
tests/                metrics, split leakage, baselines, data preparation
```

Later stages will add `retrieval/` (two-tower, FAISS), `ranking/` (candidates, LightGBM) and
`serving/` (FastAPI). Their dependencies are already declared as extras:
`uv sync --extra baselines --extra models --extra serve`.

## Evaluation rules

- A split is one cutoff date. Models and features see only transactions before it.
- Validation is the second-to-last week, test is the last week. No random splits.
- Customers evaluated are those with a purchase in the target week.
- The new-items slice is the set of articles with no sale before the cutoff.

## Future work

A/B testing simulator, PostgreSQL, Kubernetes, Airflow, streaming with Kafka.
