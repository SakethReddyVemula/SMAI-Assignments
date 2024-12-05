# Import necessary libraries
import numpy as np
import time
from sklearn.metrics import silhouette_score, davies_bouldin_score
from ..models.kmeans.kmeans import K_Means_model

class Evaluator:
    def __init__(self, Kmeans: K_Means_model):
        self.Kmeans = Kmeans

    def evaluate(self, X: np.ndarray):
        start_time = time.time()
        self.Kmeans.fit(X)
        fit_time = time.time() - start_time
        # print("fit_time: ", fit_time)
        start_time = time.time()
        labels = self.Kmeans.predict(X)
        predict_time = time.time() - start_time

        cost = self.Kmeans.getCost(X)
        silhouette = self.calculate_silhouette_score(X, labels)
        davies_bouldin = self.calculate_davies_bouldin_score(X, labels)

        return {
            'cost': cost,
            'fit_time': fit_time,
            'predict_time': predict_time,
            'num_clusters': self.Kmeans.k,
            'cluster_sizes': self.get_cluster_sizes(labels),
            'silhouette_score': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    def calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray):
        if len(np.unique(labels)) > 1:
            return silhouette_score(X, labels)
        else:
            return -1 # invalid silhouette score for only one cluster
        
    def calculate_davies_bouldin_score(self, X: np.ndarray, labels: np.ndarray):
        if len(np.unique(labels)) > 1:
            return davies_bouldin_score(X, labels)
        else:
            return -1 # invalid davies boulding socre for only one cluster
    
    def get_cluster_sizes(self, labels: np.ndarray) -> dict:
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def print_metrics(self, metrics: dict):
        print(f"\nK-means Clustering Results:")
        print(f"Number of Clusters: {metrics['num_clusters']}")
        print(f"Cost (WCSS): {metrics['cost']:.4f}")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"Davies Bouldin Score: {metrics['davies_bouldin']:.4f}")
        print(f"Fit time: {metrics['fit_time']:.4f} seconds")
        print(f"Predict time: {metrics['predict_time']:.4f} seconds")
        print(f"Cluster sizes:")
        for cluster, size in metrics['cluster_sizes'].items():
            print(f"\tCluster {cluster + 1}: {size} points")