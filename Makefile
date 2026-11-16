COMPETITION := h-and-m-personalized-fashion-recommendations
RAW := data/raw
FILES := transactions_train.csv articles.csv customers.csv

.PHONY: setup download prepare prepare-sample baselines test lint

setup:  ## Install core + dev dependencies
	uv sync --extra baselines

download:  ## Download the H&M tables (~3.5 GB unzipped; images are not fetched)
	mkdir -p $(RAW)
	for f in $(FILES); do \
		uv run --env-file .env kaggle competitions download -c $(COMPETITION) -f $$f -p $(RAW) || exit 1; \
		if [ -f $(RAW)/$$f.zip ]; then unzip -o $(RAW)/$$f.zip -d $(RAW) && rm $(RAW)/$$f.zip; fi; \
	done

prepare:  ## CSV -> Parquet, full data
	uv run python -m ecomrec.data.prepare

prepare-sample:  ## CSV -> Parquet, 10% of customers (for fast iteration)
	uv run python -m ecomrec.data.prepare --name sample --sample-frac 0.1

baselines:  ## Evaluate the baseline ladder. Override: make baselines DATASET=sample WINDOW=val
	uv run python -m ecomrec.baselines.run --dataset $(or $(DATASET),full) --window $(or $(WINDOW),val)

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .
