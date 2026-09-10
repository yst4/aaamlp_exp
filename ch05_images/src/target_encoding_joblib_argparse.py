# $ time uv run python ch05_images/src/target_envoding_joblib_argparse.py --max_depth 7 --n_estimator 200
# Fold = 0, AUC = 0.9305518815191586
# Fold = 2, AUC = 0.9334464368271581
# Fold = 1, AUC = 0.9343012918152699
# Fold = 3, AUC = 0.920384518340103
# Fold = 4, AUC = 0.9252790049575657
# Mean AUC: 0.92879

# real	0m6.948s
# user	0m29.424s
# sys	0m8.733s
# $ time uv run python ch05_images/src/target_envoding_joblib_argparse.py
# Fold = 0, AUC = 0.9314812838140261
# Fold = 3, AUC = 0.920338918334324
# Fold = 1, AUC = 0.9343099990299517
# Fold = 2, AUC = 0.9352299323855756
# Fold = 4, AUC = 0.9264818260435244
# Mean AUC: 0.92957

# real	0m8.227s
# user	0m24.844s
# sys	0m7.583s

import argparse
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

def process_fold(df, fold, cat_cols, xgb_params):
    """1つの Fold に対する Target Encoding、学習、評価を実行する関数"""
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    for col in cat_cols:
        mapping_dict = df_train.groupby(col)["income"].mean().to_dict()
        df_train[f"{col}_enc"] = df_train[col].map(mapping_dict)
        df_valid[f"{col}_enc"] = df_valid[col].map(mapping_dict)

    features = [f for f in df_train.columns if f not in ("kfold", "income")]

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # 受け取ったハイパーパラメータを展開してモデル生成
    model = xgb.XGBClassifier(
        n_jobs=1,
        random_state=42,
        **xgb_params
    )
    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")
    return auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost training across folds.")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for XGBoost")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for XGBoost")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    args = parser.parse_args()

    # パラメータを辞書化して関数に渡す
    xgb_params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
    }

    df = pd.read_csv("./input/adult_folds.csv")
    df, cat_cols = preprocess_data(df)

    results = Parallel(n_jobs=-1)(
        delayed(process_fold)(df, fold, cat_cols, xgb_params) for fold in range(5)
    )

    print(f"Mean AUC: {sum(results) / len(results):.5f}")
