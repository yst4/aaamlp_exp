# how to use the script:
# $ uv run python ch05_images/src/lbl_rf.py
# Fold = 4, AUC = 0.71825
# Fold = 2, AUC = 0.71795
# Fold = 1, AUC = 0.71764
# Fold = 3, AUC = 0.71900
# Fold = 0, AUC = 0.71768
# Mean AUC = 0.71810

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import preprocessing
from sklearn import ensemble

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

    model = ensemble.RandomForestClassifier(n_jobs=-1)

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
