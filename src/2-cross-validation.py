import pandas as pd
from sklearn import tree
from sklearn import metrics

df = pd.read_csv("input/winequality-red.csv")

# origin code: quality_mapping = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5}

quality_mapping = { i+3: i for i in range(6)}

df.loc[:, "quality"] = df.quality.map(quality_mapping)

# This is another solution of quality mapping
#
# unique_qualities = sorted(df["quality"].unique())
# quality_mapping = {val: i for i, val in enumerate(unique_qualities)}
#
# df["quality"] = df["quality"].map(quality_mapping)

# Normally train_test_split is used to split data.
# However, this manual approach is for educational purposes.
#
# frac = 1 : random choosen
# drop = True : drop an old index
df = df.sample(frac=1).reset_index(drop=True)

# total number of data;
# 1599 upper 1000 is for train and the other is for test
df_train = df.head(1000)
df_test = df.tail(599)

# another example: 80% data is for train another is for test.
# train_size = int(len(df) * 0.8)

# df_train = df.iloc[:train_size]
# df_test = df.iloc[train_size:]

clf = tree.DecisionTreeClassifier(max_depth=3)

# original code : cols is the below.
#
# cols = ['fixed acidity',
#    'volatile acidity',
#    'citric acid',
#    'residual sugar',
#    'chlorides',
#    'free sulfur dioxide',
#    'total sulfur dioxide',
#    'density',
#    'pH',
#    'sulphates',
#    'alcohol']

cols = df.columns.drop("quality")

clf.fit(df_train[cols], df_train.quality); # ; is for canceling to display the result. its too long!
