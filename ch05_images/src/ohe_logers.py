# how to use the script:
# $ uv run python ch05_images/src/ohe_logers.py
# Fold = 2, AUC = 0.78714
# Fold = 0, AUC = 0.78498
# Fold = 3, AUC = 0.78600
# Fold = 1, AUC = 0.78674
# Fold = 4, AUC = 0.78555
# Mean AUC = 0.78608

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from joblib import Parallel, delayed

def run(fold):
    df = pd.read_csv("./input/cat_train_folds.csv")

    features = [
        f for f in df.columns  if f not in ("id", "target", "kfold")
    ]

    for col in features:
        # original code: df.loc[:, col] = df[col].astype(str).fillna("NONE")
        df[col] = df[col].fillna("NONE").astype(str)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )

    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression(max_iter=1000)
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
