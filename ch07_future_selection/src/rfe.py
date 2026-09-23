import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data.data
col_names = data.feature_names
y = data.target

model = LinearRegression()
rfe = RFE(
    estimator=model,
    n_features_to_select=3
)

rfe.fit(X,y)

X_transformed = rfe.transform(X)


"""
## another solution
data = fetch_california_housing()

# pandasのデータフレーム化からRFEの適用までを宣言的に記述
X_transformed = (
    pd.DataFrame(data.data, columns=data.feature_names)
    .pipe(lambda df: RFE(LinearRegression(), n_features_to_select=3).fit_transform(df, data.target))
)
"""
