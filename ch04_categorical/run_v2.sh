#!/bin/sh
uv run python src/train_v2.py --fold 0
uv run python src/train_v2.py --fold 1
uv run python src/train_v2.py --fold 2
uv run python src/train_v2.py --fold 3
uv run python src/train_v2.py --fold 4
