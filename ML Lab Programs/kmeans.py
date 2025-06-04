import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(data.data)

kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

print("Confusion Matrix:\n", confusion_matrix(data.target, y_kmeans))
print("\nClassification Report:\n", classification_report(data.target, y_kmeans))

X_pca = PCA(n_components=2).fit_transform(X_scaled)
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = y_kmeans
df['True Label'] = data.target

def plot_clusters(title, hue, palette, centroids=False):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=hue, palette=palette, s=100, edgecolor='black', alpha=0.7)
    if centroids:
        centers = PCA(n_components=2).fit(X_scaled).transform(kmeans.cluster_centers_)
        plt.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title=hue)
    plt.show()

plot_clusters('K-Means Clustering', 'Cluster', 'Set1')
plot_clusters('True Labels', 'True Label', 'coolwarm')
plot_clusters('K-Means with Centroids', 'Cluster', 'Set1', centroids=True)
