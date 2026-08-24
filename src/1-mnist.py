import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns 

from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml(
    'mnist_784', 
    version=1, 
    return_X_y=True, 
    parser='auto'
)

pixel_values, targets = data
targets = targets.astype(int)

single_image = pixel_values.iloc[1,:].values.reshape(28,28)
plt.imshow(single_image, cmap='gray')
# plt.show() # for ipython or cli requires pyside6
