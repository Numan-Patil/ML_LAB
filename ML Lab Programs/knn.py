import numpy as np
from collections import Counter

data = np.random.rand(100)
train_data, test_data = data[:50], data[50:]
train_labels = ["Class1" if x <= 0.5 else "Class2" for x in train_data]

def knn(x, X, y, k):
    dists = sorted([(abs(x - xi), yi) for xi, yi in zip(X, y)])
    return Counter(label for _, label in dists[:k]).most_common(1)[0][0]

k_vals = [1, 2, 3, 4, 5, 20, 30]
results = {k: [knn(x, train_data, train_labels, k) for x in test_data] for k in k_vals}

for k in k_vals:
    print(f"\n--- k = {k} ---")
    for i, (x, label) in enumerate(zip(test_data, results[k]), start=51):
        print(f"x{i} = {x:.4f} -> {label}")
                