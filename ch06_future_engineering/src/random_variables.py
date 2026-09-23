import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.DataFrame(
    np.random.rand(100,2),
    columns= [f"f_{i}" for i in range(1,3)]
)

pf = preprocessing.PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=False
)

pf.fit(df)

# poly_feats = pf.fit_transform(df)
#
# num_feats = poly_feats.shape[1]
#
# df_transformed = pd.DataFrame(
#     poly_feats,
#     columns= [f"f_{i}" for i in range(1, num_feats +1)]
# )

df_transformed = pd.DataFrame(
    pf.fit_transform(df),
    columns=pf.get_feature_names_out(df.columns)
)
df_transformed.head()
df["f_bin_10"] = pd.cut(df["f_1"], bins=10, labels=False)
df["f_bin_100"] = pd.cut(df["f_1"], bins=100, labels=False)

df.head()
df.f_2.apply(lambda x: np.log(1 + x)).var()
df.f_2.var()
df.head()
