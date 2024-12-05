# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

class K_Means_model:
    def __init__(self, k: int = 3, max_iterations: int = 100, plot_fitting_graph: bool=False):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
        self.cost = None
        self.plot_fitting_graph = plot_fitting_graph

    def get_k(self):
        return self.k
    
    def set_k(self, k):
        self.k = k

    def get_max_iterations(self):
        return self.max_iterations

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations

    def fit(self, X: np.ndarray):
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        if self.plot_fitting_graph == True:
            iterations = []
            costs = []

        for iter in range(self.max_iterations):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis = 2)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            if self.plot_fitting_graph == True:
                iterations.append(iter)
                costs.append(self.getCost(X))
            
            if np.all(self.centroids == new_centroids):
                # print(f"Break at {iter}")
                break
            # print(self.getCost(X))
            

            self.centroids = new_centroids

        if self.plot_fitting_graph == True:
            plt.figure(figsize=(8, 8))
            plt.plot(iterations, costs, marker='o')
            plt.title(f"Fitting curve of K-means clustering; k={self.k}")
            plt.xlabel("iterations")
            plt.ylabel("WCSS cost")
            plt.savefig("assignments/2/figures/kmeans.png")

        self.cost = self.getCost(X)

    def getCost(self, X: np.ndarray) -> float:
        distances = np.min(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2), axis=1)
        return np.sum(distances)
    
    def predict(self, X: np.ndarray):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
