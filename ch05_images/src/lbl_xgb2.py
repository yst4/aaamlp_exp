# how to use the script:
# $ uv run python ch05_images/src/lbl_xgb2.py
# Fold = 3, AUC = 0.86634
# Fold = 1, AUC = 0.88502
# Fold = 2, AUC = 0.88047
# Fold = 0, AUC = 0.87533
# Fold = 4, AUC = 0.87109
# Mean AUC = 0.87565
#
# uv run python ch05_images/src/lbl_xgb2.py --max_depth 7 --n_estimators 200
# Fold = 0, AUC = 0.86605
# Fold = 2, AUC = 0.87342
# Fold = 3, AUC = 0.86000
# Fold = 1, AUC = 0.87726
# Fold = 4, AUC = 0.86114
# Mean AUC = 0.86757

import pandas as pd
import xgboost as xgb

from sklearn import preprocessing, metrics
import numpy as np

from joblib import Parallel, delayed
import argparse

def run(fold, max_depth=6, n_estimators=100):
    df = pd.read_csv("./input/adult_folds.csv")

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

    df = df.drop(num_cols, axis=1)

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df["income"] = df.income.map(target_mapping)

    features = [
        f for f in df.columns  if f not in ("income", "kfold")
    ]

    for col in features:
        # original code: df.loc[:, col] = df[col].astype(str).fillna("NONE")
        df[col] = df[col].fillna("NONE").astype(str)

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])

        df[col] = lbl.transform(df[col])


    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=max_depth,
        n_estimators=n_estimators
    )
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc:.5f}")
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_depth",
        type=int,
        default=6
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )
    args = parser.parse_args()
#    for fold_ in range(5):
#        run(fold_)
    results = Parallel(n_jobs=-1)(
        delayed(run)(fold_, max_depth=args.max_depth, n_estimators=args.n_estimators) for fold_ in range(5)
    )

    # 各foldの結果リストを受け取って平均を算出
    print(f"Mean AUC = {np.mean(results):.5f}")
