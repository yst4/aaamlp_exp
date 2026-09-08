# how to use the script:
# $ uv run python ch05_images/src/lbl_xgb_feat.py
# Fold = 0, AUC = 0.92623
# Fold = 2, AUC = 0.92968
# Fold = 1, AUC = 0.93029
# Fold = 3, AUC = 0.91660
# Fold = 4, AUC = 0.92223
# Mean AUC = 0.92501

import pandas as pd
import xgboost as xgb

from sklearn import preprocessing, metrics
import numpy as np

import itertools
from joblib import Parallel, delayed
import argparse

def feature_engineering(df, cat_cols):
    """
    This function is used for feature engineering
    Args:
        df: the pandas dataframe with train/test data
        cat_cols: list of categorical columns
    Results:
        dataframe with new features
    """

    combi = itertools.combinations(cat_cols,2)

    for c1,c2 in combi:
        df[c1 + " " + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df

def run(fold, max_depth=6, n_estimators=100):
    df = pd.read_csv("./input/adult_folds.csv")

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

  #  df = df.drop(num_cols, axis=1)

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df["income"] = df.income.map(target_mapping)

    # modify: remove if clause of original code.
    cat_cols = [
        c for c in df.columns if c not in num_cols + ["income", "kfold"]
    ]

    df = feature_engineering(df, cat_cols)

    features = [
        f for f in df.columns  if f not in ("income", "kfold")
    ]

    cat_cols_all = [c for c in features if c not in num_cols]

    for col in cat_cols_all:
        df[col] = df[col].fillna("NONE").astype(str)
        lbl = preprocessing.LabelEncoder()
        df[col] = lbl.fit_transform(df[col])


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
