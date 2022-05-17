import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#read the previously cleaned csv
df_pivot = pd.read_csv('pivot.csv')
df_pivot = StandardScaler().fit_transform(df_pivot)
df_pivot = pd.DataFrame(df_pivot)
df_pivot = df_pivot.fillna(0)
pca = PCA(n_components = 3, random_state = 7)
pca_mdl = pca.fit_transform(df_pivot)
pca_df = pd.DataFrame(pca_mdl)
kmeans = KMeans(n_clusters=8, max_iter=1000, algorithm='auto')
fit_result = kmeans.fit(pca_df)
preds=kmeans.predict(pca_df)


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

x = pca_df[0]
y = pca_df[1]
z = pca_df[2]

ax.set_xlabel("PC 1") # principle component #1
ax.set_ylabel("PC 2") # principle component #2
ax.set_zlabel("PC 3") # principle component #3

ax.scatter(x, y, z, c = preds)

plt.show()

