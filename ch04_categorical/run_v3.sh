#!/bin/sh
uv run python src/train_v3.py --fold 0 --model rf
uv run python src/train_v3.py --fold 1 --model rf
uv run python src/train_v3.py --fold 2 --model rf
uv run python src/train_v3.py --fold 3 --model rf
uv run python src/train_v3.py --fold 4 --model rf
