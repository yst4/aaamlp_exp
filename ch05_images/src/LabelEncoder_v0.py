import pandas as pd

categories = ["Freezing", "warm", "Cold", "Boiling Hot", "Hot", "Lava Hot"]
mapping ={ k:v for (v,k) in enumerate(categories) }

# mapping = {'Freezing': 0,
#  'warm': 1,
#  'Cold': 2,
#  'Boiling Hot': 3,
#  'Hot': 4,
#  'Lava Hot': 5}

df = pd.read_csv("./input/cat_train.csv")
df.ord_2.value_counts()

# df.loc[:, "ord_2"] = df["ord_2"].map(mapping)
df["ord_2"] = df["ord_2"].map(mapping)
df.head()
df.ord_2.value_counts()
