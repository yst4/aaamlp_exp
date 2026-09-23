import pandas as pd
import numpy as np
import matplotlib.pylab as plt


from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

data = load_diabetes()
X= data.data
col_names = data.feature_names
y = data.target

model = RandomForestRegressor()

model.fit(X,y)
importances = model.feature_importances_
idxs = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(idxs)), importances[idxs], align='center')
plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
plt.xlabel('Random Forest Feature Importance')
plt.show()
