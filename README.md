# EComRec

Two-stage e-commerce recommender: two-tower retrieval + FAISS + LightGBM ranking on 31M H&M
transactions, with multimodal cold-start and a FastAPI service under 50 ms p99.

**Status:** Week 2 done. The baseline ladder and the two-tower retriever (with a FAISS index
and MLflow tracking) are measured on the full data. Next is the ranking stage.

## Results

Full data, time-based holdout. Test week is 16–22 September 2020: 68,984 customers with a
purchase, and 667 articles that had never sold before.

| Model | Recall@100 | NDCG@12 | MAP@12 | New-item Recall@100 |
| --- | --- | --- | --- | --- |
| Popularity (last 7 days) | 0.1181 | 0.0158 | 0.0087 | 0.0000 |
| Repurchase + popular | 0.1224 | 0.0341 | 0.0247 | 0.0000 |
| Item-to-item co-purchase | 0.1285 | 0.0361 | 0.0245 | 0.0000 |
| ALS | 0.1424 | 0.0321 | 0.0207 | 0.0000 |
| Two-tower only | 0.1523 | 0.0264 | 0.0155 | 0.0000 |
| Two-tower + ranker | – | – | – | – |
| + content item tower | – | – | – | – |

What the results show so far:

- **Repurchase is hard to beat at the top of the list.** Co-purchase edges it on NDCG@12, but
  the 95% bootstrap intervals nearly touch (0.0351–0.0371 vs 0.0331–0.0352) and MAP@12 is level.
- **ALS is the best retriever and the worst ranker of the three personalized baselines.** It has
  the highest Recall@100 but lower NDCG@12 than plain repurchase. That is the gap a ranking
  stage is meant to close.
- **No baseline can recommend a new product.** New-item recall is zero by construction, which is
  the starting point for the content-based item tower.
- **Recency matters more than volume.** Co-purchase and ALS use only the last 28 days of
  history. On the validation week that beat 56 days and longer for both models.
- **The two-tower model is the best retriever, and not yet a good ranker.** It beats ALS on
  Recall@100 by 7% (0.1523 vs 0.1424) but its NDCG@12 is below every personalized baseline.
  It is a candidate generator; ordering the top 12 is the ranker's job.
- **The logQ correction is not optional.** Without it, in-batch negatives punish popular items
  and Recall@100 collapsed from about 0.07 to 0.015 on the 10% sample.
- **Recency helped the neural model as much as the baselines.** Training on 28 days instead of
  56, indexing only articles sold in the last 14 days, and telling the user tower how long ago
  the customer last bought took validation Recall@100 from 0.119 to 0.143.

Tuning used the validation week only: the baselines' history window and ALS `alpha=40`, and
the two-tower training window, epoch count and index window. The test week was run once per
model. Each number is a single run (one seed), and only NDCG@12 and MAP@12 have confidence
intervals so far.

Reproduce with `make baselines WINDOW=test` and `make two-tower WINDOW=test`. Both also print
confidence intervals, catalog coverage and popularity bias.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and a Kaggle account that has accepted the
[competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).

```bash
cp .env.example .env    # then add your Kaggle API token
make setup              # install dependencies (Python 3.12 is fetched by uv)
make download           # the three CSV tables, about 3.5 GB; images are skipped for now
make prepare-sample     # CSV -> Parquet for 10% of customers
make baselines DATASET=sample WINDOW=val
make two-tower DATASET=sample WINDOW=val
make test
```

`make prepare`, `make baselines` and `make two-tower` do the same on the full data. On a
laptop CPU the full baseline run takes about a minute and two-tower training about three.

`make two-tower` trains the model, logs the run to MLflow (`uv run mlflow ui` to browse it),
exports embeddings to `artifacts/`, then evaluates retrieval again through a FAISS index in a
separate process. torch and faiss cannot share a process on macOS, so training and indexing
stay apart, as they will in serving.

## Layout

```
src/ecomrec/
  config.py           paths and constants (k=12, 7-day windows)
  data/prepare.py     raw CSV -> typed Parquet, optional customer sampling
  data/split.py       cutoff-date splits, val/test windows, new-items slice
  eval/metrics.py     Recall@k, MAP@k (Kaggle definition), NDCG@k, bootstrap CI
  eval/evaluate.py    scores a model's recommendations on a split
  eval/report.py      window loading and results-table rows for the CLIs
  baselines/          popularity, repurchase, co-purchase, ALS, and the runner
  retrieval/
    features.py       item attribute codes, leakage-safe purchase-history windows
    model.py          two-tower model, in-batch softmax loss with logQ correction
    two_tower.py      training, inference and embedding export
    train.py          training CLI with MLflow tracking
    index.py          FAISS index and its evaluation CLI
tests/                metrics, split leakage, baselines, retrieval, data preparation
```

Later stages will add `ranking/` (candidates, LightGBM) and `serving/` (FastAPI). The serving
dependencies are already declared as an extra: `uv sync --extra serve`.

## Evaluation rules

- A split is one cutoff date. Models and features see only transactions before it.
- Validation is the second-to-last week, test is the last week. No random splits.
- Customers evaluated are those with a purchase in the target week.
- The new-items slice is the set of articles with no sale before the cutoff.

## Future work

A/B testing simulator, PostgreSQL, Kubernetes, Airflow, streaming with Kafka.
