import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("./input/cat_train.csv")
df[df.ord_2 == "Boiling Hot"].shape
df.groupby(["ord_2"])["id"].count()
df.groupby(["ord_2"])["id"].transform("count")
df.groupby(
    [
        "ord_1",
        "ord_2"
    ]
)["id"].count().reset_index(name="count")
df["new_feature"] = (
    df.ord_1.astype(str)
    + "_"
    + df.ord_2.astype(str)
)
df.new_feature
df["new_feature"] = (
    df.ord_1.astype(str)
    + "_"
    + df.ord_2.astype(str)
    + "_"
    + df.ord_3.astype(str)
)
df.new_feature
df.ord_2.value_counts()
df.ord_2.fillna("NONE").value_counts()
