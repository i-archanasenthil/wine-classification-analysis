import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
print(wine_df.dtypes)
print(wine_df.isna().sum())

X = wine_df.drop(columns = ['target'], errors = 'ignore')
y = wine_df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data= X_pca, columns=['PC1','PC2'])
pca_df['target'] = y

plt.figure(figsize = (10,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target',palette={0:'red', 1:'green',2:'blue'})
plt.title('PCA of Wine Classification')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()