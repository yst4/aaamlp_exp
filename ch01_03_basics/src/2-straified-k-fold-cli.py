#!/usr/bin/env python3

# Note: Requires Python 3.10 or higher (uses `|` union type hints)
#
# Created with Gemini.
# Run with: python src/2-straitfied-k-fold-cli.py --help

import argparse
import pandas as pd
from sklearn import model_selection


def create_folds(input_path: str, output_path: str, random_state: int | None):
    df = pd.read_csv(input_path)
    df["kfold"] = -1

    if random_state is not None:
        kf = model_selection.StratifiedKFold(
            n_splits=5, shuffle=True, random_state=random_state
        )
    else:
        kf = model_selection.StratifiedKFold(n_splits=5)

    y = df.target.values

    for fold, (_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, "kfold"] = fold

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified K-Fold を作成して出力します。"
    )
    parser.add_argument("input_csv", type=str, help="入力CSVファイルパス")
    parser.add_argument("output_csv", type=str, help="出力CSVファイルパス")
    parser.add_argument(
        "--rand-seed",
        type=int,
        default=None,
        help="乱数シード（指定すると shuffle=True になります）",
    )

    args = parser.parse_args()

    create_folds(
        input_path=args.input_csv,
        output_path=args.output_csv,
        random_state=args.rand_seed,
    )
