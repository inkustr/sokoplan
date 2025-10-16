.venv: ; python -m venv .venv
lint: ; ruff check . && black --check . && mypy sokoban_core heuristics search gnn
test: ; pytest -q
train: ; python scripts/train_gnn.py
search: ; python scripts/run_search.py
