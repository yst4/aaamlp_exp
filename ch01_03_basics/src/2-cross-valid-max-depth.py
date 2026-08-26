import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, tree

### helper functions

def evaluate_train_and_test(
    model, train_df, test_df, features, target_col="quality"
):
    """return a pair of accuracy scores for train and test data"""
    train_acc = metrics.accuracy_score(
        train_df[target_col], model.predict(train_df[features])
    )
    test_acc = metrics.accuracy_score(
        test_df[target_col], model.predict(test_df[features])
    )
    return train_acc, test_acc

### main part

df = pd.read_csv("input/winequality-red.csv")

unique_qualities = sorted(df["quality"].unique())
quality_mapping = {val: i for i, val in enumerate(unique_qualities)}

df.loc[:, "quality"] = df.quality.map(quality_mapping)
cols = df.columns.drop("quality")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_train = df.head(1000)
df_test = df.tail(599)

results_acc = [(0, 0.5, 0.5)]  # (depth, train_accuracy, test_accuracy)
max_depth = 25

for depth in range(1, max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(df_train[cols], df_train.quality)

    train_acc, test_acc = evaluate_train_and_test(
        clf, df_train, df_test, cols
    )

    results_acc.append((depth, train_acc, test_acc))

### depict
depths, train_accs, test_accs = zip(*results_acc)

matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)

plt.figure(figsize=(10, 6))
plt.plot(depths, train_accs, label="train accuracy")
plt.plot(depths, test_accs, label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()
