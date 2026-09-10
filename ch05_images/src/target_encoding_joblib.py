# $ time uv run python ch05_images/src/target_encoding_joblib.py
# Fold = 1, AUC = 0.9298613863020937
# Fold = 3, AUC = 0.9180913604822997
# Fold = 2, AUC = 0.9300365625619179
# Fold = 4, AUC = 0.9231109085017501
# Fold = 0, AUC = 0.9279796048438496
# Mean AUC: 0.92582

# real	0m6.335s
# user	0m24.187s
# sys	0m8.663s

import copy
import pandas as pd
from sklearn import metrics, preprocessing
import xgboost as xgb
from joblib import Parallel, delayed

def preprocess_data(data):
    df = copy.deepcopy(data)
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    target_mapping = {"<=50K": 0, ">50K": 1}

    df["income"] = df["income"].map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income")]
    cat_cols = [c for c in features if c not in num_cols]

    for col in cat_cols:
        df[col] = df[col].fillna("NONE").astype(str)
        lbl = preprocessing.LabelEncoder()
        df[col] = lbl.fit_transform(df[col])

    return df, cat_cols

def process_fold(df, fold, cat_cols):
    """1つの Fold に対する Target Encoding、学習、評価を実行する関数"""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Train データから Target Encoding の辞書を作成して Valid に適用
    for col in cat_cols:
        mapping_dict = df_train.groupby(col)["income"].mean().to_dict()
        df_train[f"{col}_enc"] = df_train[col].map(mapping_dict)
        df_valid[f"{col}_enc"] = df_valid[col].map(mapping_dict)

    features = [f for f in df_train.columns if f not in ("kfold", "income")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # 各プロセス内でのスレッド競合を防ぐため n_jobs=1 を推奨
    model = xgb.XGBClassifier(n_jobs=1)
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")
    return auc

if __name__ == "__main__":
    # 1. データの読み込み
    df = pd.read_csv("./input/adult_folds.csv")

    # 2. 前処理（カテゴリ補正やラベルエンコーディング等）
    df, cat_cols = preprocess_data(df)

    # 3. 5 Fold の学習・評価を並列実行
    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(df, fold, cat_cols) for fold in range(5)
    )
    print(f"Mean AUC: {sum(results) / len(results):.5f}")
