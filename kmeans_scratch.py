import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=300):
        self.k=k
        self.max_iters=max_iters

    def fit(self, X):
        n_samples, n_features = X.shape
        rng = np.random.default_rng(42)
        self.centroids = X[rng.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) 
                                      else self.centroids[i] for i in range(self.k)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)
