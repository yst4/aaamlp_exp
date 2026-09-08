# how to use the script:
# $ uv run python ch05_images/src/ohe_logers2.py
# Fold = 3, AUC = 0.86818
# Fold = 2, AUC = 0.88521
# Fold = 0, AUC = 0.87950
# Fold = 1, AUC = 0.88749
# Fold = 4, AUC = 0.87294
# Mean AUC = 0.87866

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import linear_model, metrics, preprocessing


def run(fold):
    df = pd.read_csv("./input/adult_folds.csv")
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ]

    df = df.drop(num_cols, axis=1)

    target_mapping = {"<=50K": 0, ">50K": 1}
    df["income"] = df.income.map(target_mapping)
    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        df[col] = df[col].fillna("NONE").astype(str)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 訓練データのみでfitする
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")
    x_train = ohe.fit_transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    # 分類モデルを使用
    model = linear_model.LogisticRegression(max_iter=1000)
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    # 正しい列名 (income) で評価
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc:.5f}")
    return auc


if __name__ == "__main__":
    results = Parallel(n_jobs=-1)(delayed(run)(fold_) for fold_ in range(5))
    print(f"Mean AUC = {np.mean(results):.5f}")
