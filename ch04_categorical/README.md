## :warning: Caution: Input Dataset Setup

This repository is a sub-project of `aaamlp_exp` and depends on the dataset from **AAAMLP** (Approaching (Almost) Any Machine Learning Problem).

### Data Location & Symbolic Links
The required dataset should be placed in `ch04_categorical/input/`. You can create a symbolic link from `aaamlp_exp/input/` to `ch04_categorical/input/`.

> **Note:** If you use relative paths for symbolic links, make sure they are relative to the `input/` directory, not the project root.

### Running Scripts
You can run the scripts using `uv` from either project context:

```bash
# From inside ch04_categorical
uv run python src/train.py

# From the root project (aaamlp_exp)
uv run python ch04_categorical/src/train.py
```
