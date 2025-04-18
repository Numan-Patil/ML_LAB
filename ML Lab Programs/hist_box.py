import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing as fch

df = fch(as_frame=True).frame
num_cols = df.select_dtypes(include=np.number).columns

df.hist(figsize=(15, 10), bins=30, color='skyblue')
plt.suptitle('Histograms of Numerical Features', fontsize=16)
plt.tight_layout(); plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=df[col], color='orange')
    plt.title(col)
plt.tight_layout(); plt.show()

print("Outliers Detection:")
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

print("\nDataset Summary:")
print(df.describe())
