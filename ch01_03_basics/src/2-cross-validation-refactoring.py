import pandas as pd
from sklearn import tree, metrics

df = pd.read_csv("input/winequality-red.csv")

# ターゲット値のマッピング
quality_mapping = {i + 3: i for i in range(6)}
df.loc[:, "quality"] = df.quality.map(quality_mapping)

# データのシャッフルと分割
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_train = df.head(1000)
df_test = df.tail(599)

# 特徴量列の指定とモデル学習
cols = df.columns.drop("quality")
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(df_train[cols], df_train.quality)

# ヘルパー関数定義
def evaluate_train_and_test(model, train_df, test_df, features, target_col="quality"):
    """return a pair of accuracy score of train-and-test data"""
    train_acc = metrics.accuracy_score(
        train_df[target_col], model.predict(train_df[features])
    )
    test_acc = metrics.accuracy_score(
        test_df[target_col], model.predict(test_df[features])
    )
    return train_acc, test_acc

# 評価実行
train_acc, test_acc = evaluate_train_and_test(clf, df_train, df_test, cols)
print(f"Accuracy_score > train : {train_acc:.4f}, test : {test_acc:.4f}")
