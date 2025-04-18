import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as fch

df = fch(as_frame=True).frame

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()