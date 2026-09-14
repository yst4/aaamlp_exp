import numpy as np
from sklearn import impute

X = np.random.randint(1,15,(10,6))

X = X.astype(float)

X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan
knn_imputer = impute.KNNImputer(n_neighbors=2)
knn_imputer.fit_transform(X)
