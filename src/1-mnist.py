from seaborn.axisgrid import FacetGrid
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
    parser='auto' # require declaration parser setting for current update
)

pixel_values, targets = data
targets = targets.astype(int)

# Origin code: pixel_values.reshape(28,28) -> pandas Series needs array conversion
single_image = pixel_values.iloc[1].to_numpy().reshape(28, 28)

plt.imshow(single_image, cmap='gray')
# plt.show() # for ipython or cli requires pyside6

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values.iloc[:3000,:])
tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=["x","y","targets"]
)

tsne_df["targets"] = tsne_df.targets.astype(int)

grid = FacetGrid(tsne_df, hue="targets", height=8) # origin code is `size` instead of `height`
# Origin code: grid.map(plt.scatter, "x", "y").add_legend() to recommend code is the below.
grid.map_dataframe(sns.scatterplot, x="x", y="y").add_legend()
# plt.show() # for ipython or cli requires pyside6
