# how to use the script:
# $ uv run python ch05_images/src/lbl_xgb.py
# Fold = 1, AUC = 0.75982
# Fold = 2, AUC = 0.76213
# Fold = 0, AUC = 0.75973
# Fold = 3, AUC = 0.75909
# Fold = 4, AUC = 0.76030
# Mean AUC = 0.76021

import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, metrics
import numpy as np
from joblib import Parallel, delayed

def run(fold):
    df = pd.read_csv("./input/cat_train_folds.csv")

    features = [
        f for f in df.columns  if f not in ("id", "target", "kfold")
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
        max_depth=7,
        n_estimators=200
    )
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc:.5f}")
    return auc

if __name__ == "__main__":
#    for fold_ in range(5):
#        run(fold_)
    results = Parallel(n_jobs=-1)(
        delayed(run)(fold_) for fold_ in range(5)
    )

    # 各foldの結果リストを受け取って平均を算出
    print(f"Mean AUC = {np.mean(results):.5f}")
