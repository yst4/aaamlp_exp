import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("./input/cat_train.csv")

df.ord_2.fillna("NONE").value_counts()
df.ord_4.fillna("NONE").value_counts()

# original line: (for old pandas)
# df.loc[
#  df["ord_4"].value_counts()[df["ord_4"]].values < 2000,
#  "ord_4"
# ] = "RARE"

# for current pandas
counts = df["ord_4"].value_counts()
df.loc[df["ord_4"].map(counts) < 2000, "ord_4"] = "RARE"

# yet another approach.
df.loc[df.groupby("ord_4")["ord_4"].transform("count") < 2000, "ord_4"] = "RARE"
