import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
pca_data = PCA(n_components=2).fit_transform(iris.data)

df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df['Label'] = iris.target

plt.figure(figsize=(8, 6))
for i, name in enumerate(iris.target_names):
    plt.scatter(df[df.Label == i]['PC1'], df[df.Label == i]['PC2'], label=name)

plt.title('PCA on Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid()
plt.show()