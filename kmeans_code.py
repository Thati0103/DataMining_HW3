import numpy as np
import pandas as pd
from collections import Counter


def euclidean_vectorized(X, C):
    return np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))

def cosine_vectorized(X, C):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-10)
    return 1 - np.dot(Xn, Cn.T)

def jaccard_vectorized(X, C):
    mins = np.minimum(X[:, None, :], C[None, :, :]).sum(axis=2)
    maxs = np.maximum(X[:, None, :], C[None, :, :]).sum(axis=2)
    return 1 - (mins / (maxs + 1e-10))

def get_distance(metric):
    if metric == "euclidean":
        return euclidean_vectorized
    elif metric == "cosine":
        return cosine_vectorized
    elif metric == "jaccard":
        return jaccard_vectorized
    else:
        raise ValueError("Unknown metric")


class KMeansFast:

    def __init__(self, n_clusters, max_iter=100, tol=1e-4,
                 metric="euclidean", stop="centroid"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.stop = stop
        self.dist_func = get_distance(metric)

    def fit(self, X):
        n, d = X.shape
        rng = np.random.default_rng()

        # random centroids
        self.centroids = X[rng.choice(n, self.n_clusters, replace=False)]

        prev_sse = None

        for it in range(self.max_iter):

            distances = self.dist_func(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                group = X[labels == k]
                if len(group) > 0:
                    new_centroids[k] = group.mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            sse = np.sum(distances[np.arange(n), labels] ** 2)

            shift = np.linalg.norm(new_centroids - self.centroids)

            if self.stop == "centroid" and shift < self.tol:
                break
            if self.stop == "sse" and prev_sse is not None and sse > prev_sse:
                break
            if self.stop == "max_iter" and it == self.max_iter - 1:
                break

            prev_sse = sse
            self.centroids = new_centroids

        # Save results
        self.labels_ = labels
        self.sse_ = sse
        self.iterations_ = it + 1


def cluster_accuracy(y, pred):
    mapping = {}
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        majority = Counter(y[idx]).most_common(1)[0][0]
        mapping[c] = majority
    mapped = np.array([mapping[c] for c in pred])
    return np.mean(mapped == y)


X = pd.read_csv("data.csv", header=None).values
y = pd.read_csv("label.csv", header=None).values.flatten()

n_clusters = len(np.unique(y))
metrics = ["euclidean", "cosine", "jaccard"]
stops = ["centroid", "sse", "max_iter"]

results = []


for metric in metrics:
    for stop in stops:

        km = KMeansFast(
            n_clusters=n_clusters,
            max_iter=100,
            metric=metric,
            stop=stop
        )

        km.fit(X)
        acc = cluster_accuracy(y, km.labels_)

        results.append([
            metric,
            stop,
            km.sse_,
            acc,
            km.iterations_
        ])

# Save results to DataFrame
df = pd.DataFrame(results, columns=[
    "Distance Metric",
    "Stop Criteria",
    "SSE",
    "Accuracy",
    "Iterations"
])

print(df)

with open("kmeans_results.txt", "w") as f:

    f.write("===== K-MEANS RESULTS (OPTIMIZED VERSION) =====\n\n")

    for metric in metrics:
        f.write(f"\n===== {metric.upper()} =====\n")
        block = df[df["Distance Metric"] == metric]

        for stop in stops:
            row = block[block["Stop Criteria"] == stop].iloc[0]
            f.write(
                f"\nStop: {stop}\n"
                f"SSE: {row['SSE']:.4f}\n"
                f"Accuracy: {row['Accuracy']:.4f}\n"
                f"Iterations: {row['Iterations']}\n"
            )

    f.write("\n===== SUMMARY TABLE =====\n")
    f.write(df.to_string(index=False))

print("\nResults saved to: kmeans_results.txt")
