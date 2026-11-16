# EComRec

Two-stage e-commerce recommender: two-tower retrieval + FAISS + LightGBM ranking on 31M H&M
transactions, with multimodal cold-start and a FastAPI service under 50 ms p99.

**Status:** Week 1 done. The data pipeline, time-based split, evaluation harness and the full
baseline ladder are in place and measured on the full data. Next is the two-tower retriever.

## Results

Full data, time-based holdout. Test week is 16–22 September 2020: 68,984 customers with a
purchase, and 667 articles that had never sold before.

| Model | Recall@100 | NDCG@12 | MAP@12 | New-item Recall@100 |
| --- | --- | --- | --- | --- |
| Popularity (last 7 days) | 0.1181 | 0.0158 | 0.0087 | 0.0000 |
| Repurchase + popular | 0.1224 | 0.0341 | 0.0247 | 0.0000 |
| Item-to-item co-purchase | 0.1285 | 0.0361 | 0.0245 | 0.0000 |
| ALS | 0.1424 | 0.0321 | 0.0207 | 0.0000 |
| Two-tower only | – | – | – | – |
| Two-tower + ranker | – | – | – | – |
| + content item tower | – | – | – | – |

What the baselines show so far:

- **Repurchase is hard to beat at the top of the list.** Co-purchase edges it on NDCG@12, but
  the 95% bootstrap intervals nearly touch (0.0351–0.0371 vs 0.0331–0.0352) and MAP@12 is level.
- **ALS is the best retriever and the worst ranker of the three personalized baselines.** It has
  the highest Recall@100 but lower NDCG@12 than plain repurchase. That is the gap a ranking
  stage is meant to close.
- **No baseline can recommend a new product.** New-item recall is zero by construction, which is
  the starting point for the content-based item tower.
- **Recency matters more than volume.** Co-purchase and ALS use only the last 28 days of
  history. On the validation week that beat 56 days and longer for both models.

The history window (and ALS `alpha=40`) were chosen on the validation week; nothing else is
tuned. Reproduce with `make baselines WINDOW=test`, which also prints confidence intervals,
catalog coverage and popularity bias.

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

`make prepare` and `make baselines` do the same on the full data. The full baseline run takes
about a minute on a laptop.

## Layout

```
src/ecomrec/
  config.py           paths and constants (k=12, 7-day windows)
  data/prepare.py     raw CSV -> typed Parquet, optional customer sampling
  data/split.py       cutoff-date splits, val/test windows, new-items slice
  eval/metrics.py     Recall@k, MAP@k (Kaggle definition), NDCG@k, bootstrap CI
  eval/evaluate.py    scores a model's recommendations on a split
  baselines/          popularity, repurchase, co-purchase, ALS, and the runner
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
