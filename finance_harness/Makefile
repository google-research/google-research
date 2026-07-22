# FinanceHarness — headless entrypoints. `make help` lists targets.
.DEFAULT_GOAL := help
.PHONY: help install run serve

# the service port (override e.g. `make serve PORT=9000`)
PORT ?= 8080

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | sort

install:  ## install dependencies
	uv sync || pip install -e .

run:  ## one-shot research: make run Q="your question"
	python main.py -p "$(Q)"

serve:  ## run the optional HTTP+SSE service on $(PORT)
	uv run fh serve --port $(PORT)
