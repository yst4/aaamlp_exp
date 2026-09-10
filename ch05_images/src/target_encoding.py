# The code modified by Gemini of Google
# $ time uv run python ch05_images/src/target_encoding.py
# Fold = 0, AUC = 0.92465
# Fold = 1, AUC = 0.92840
# Fold = 2, AUC = 0.92840
# Fold = 3, AUC = 0.91681
# Fold = 4, AUC = 0.92170
# Mean AUC: 0.92399

# real	0m7.302s
# user	0m36.365s
# sys	0m0.994s

import pandas as pd
from sklearn import metrics
import xgboost as xgb

def mean_target_encoding(df):
    df = df.copy()

    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]

    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }

    df["income"] = df["income"].astype(str).str.strip().map(target_mapping)

    cat_cols = [
        f for f in df.columns if f not in ("kfold", "income") and f not in num_cols
    ]

    encoded_dfs = []

    # FoldごとにTarget Encodingを実施してdf_validを蓄積
    for fold in range(5):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        global_mean = df_train["income"].mean()

        for column in cat_cols:
            mapping_dict = dict(df_train.groupby(column)["income"].mean())
            # Validationデータに対してのみ、Trainから計算した平均値を適用
            df_valid[column + "_enc"] = df_valid[column].map(mapping_dict).fillna(global_mean)

        encoded_dfs.append(df_valid)

    # 5つのValidationデータを結合してOOF済みの全データフレームを作成
    encoded_df = pd.concat(encoded_dfs, axis=0).reset_index(drop=True)
    return encoded_df

def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 数値列とTarget Encoding済みの列(_enc)のみを選択
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    enc_cols = [f for f in df.columns if f.endswith("_enc")]
    features = num_cols + enc_cols

    x_train = df_train[features].values
    x_valid = df_valid[features].values

    # Train側のTarget EncodingをFold内のTrainデータのみから再計算（リーク防止）
    # ※本番運用ではdf_train側もFold内でTarget Encodingを計算して適用します
    for col in [f.replace("_enc", "") for f in enc_cols]:
        global_mean = df_train["income"].mean()
        mapping_dict = dict(df_train.groupby(col)["income"].mean())
        # x_trainの該当列を上書き/変換
        col_idx = features.index(col + "_enc")
        x_train[:, col_idx] = df_train[col].map(mapping_dict).fillna(global_mean).values

    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        random_state=42
    )

    model.fit(x_train, df_train.income.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc:.5f}")
    return auc

if __name__ == "__main__":
    df = pd.read_csv("./input/adult_folds.csv")
    df = mean_target_encoding(df)

    results = []
    for fold_ in range(5):
        auc = run(df, fold_)
        results.append(auc)

    print(f"Mean AUC: {sum(results) / len(results):.5f}")
