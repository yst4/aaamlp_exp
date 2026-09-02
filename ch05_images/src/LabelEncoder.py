import pandas as pd
from sklearn import preprocessing

df = pd.read_csv("./input/cat_train.csv")

# df.loc[:, "ord_2"] = df.ord_2.fillna("NONE")  #Original code.
df["ord_2"] = df.ord_2.fillna("NONE")

lbl_enc = preprocessing.LabelEncoder()

# df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)
#   # Original code but it is error on current version.
df["ord_2"] = lbl_enc.fit_transform(df.ord_2.astype(str).values)
df.head()
