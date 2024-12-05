# Import necessary libraries
import pandas as pd
import numpy as np
from typing import List, Tuple
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Import models
from models.kmeans.kmeans import K_Means_model
from models.gmm.gmm import GMM
from models.pca.pca import PCA
from models.knn.knn import Best_KNN_model

# Import Evaluation classes
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

class Best_KNN_evaluate:
    # TC: O(1)
    def __init__(self, KNN):
        self.KNN: object = KNN
        self.validation_split = self.KNN.validation_split
        self.test_split = self.KNN.test_split
    
    # TC: O(mc)
    # SC: O(c + m)
    # calculate the evaluation scores manually using numpy
    def calculate_metrics(self, true_y, pred_y):
        unique_classes = np.unique(true_y)
        
        # Initialize dictionaries for macro scores
        precision_dict = defaultdict(float)
        recall_dict = defaultdict(float)
        F1_dict = defaultdict(float)
        
        # Initialize variables for micro scores
        tp_micro = 0
        fp_micro = 0
        fn_micro = 0
        
        for cls in unique_classes:
            tp = np.sum((true_y == cls) & (pred_y == cls))
            fp = np.sum((true_y != cls) & (pred_y == cls))
            fn = np.sum((true_y == cls) & (pred_y != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            F1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_dict[cls] = precision
            recall_dict[cls] = recall
            F1_dict[cls] = F1_score
            
            # for micro scores, accumulate tp, fp, fn
            tp_micro += tp
            fp_micro += fp
            fn_micro += fn
        
        # Calculate macro scores
        macro_precision = np.mean(list(precision_dict.values()))
        macro_recall = np.mean(list(recall_dict.values()))
        macro_F1_score = np.mean(list(F1_dict.values()))
        
        # Calculate micro scores
        micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        micro_F1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        accuracy = np.mean(true_y == pred_y)
        
        return accuracy, macro_precision, macro_recall, macro_F1_score, micro_precision, micro_recall, micro_F1_score
    
    # TC: O(m(nd + k log k) + mc)
    # SC: O(mn + c + m)
    def evaluate(self, X_test, y_test):
        # true_y = X_test['track_genre'].values
        true_y = np.array(y_test)
        pred_y = np.array(self.KNN.predict(X_test))
        
        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1 = self.calculate_metrics(true_y, pred_y)

        avg_time_taken_per_prediction = self.KNN.total_time_taken / self.KNN.prediction_count

        return {
            'accuracy': acc,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'avg_time': avg_time_taken_per_prediction
        }
    
    # TC; O(c)
    def print_metrics(self, metrics, set_name):
        print(f"\n{set_name} Set Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")




# Read the data
path_to_embeddings = "data/external/word-embeddings.feather"
df = pd.read_feather(path_to_embeddings)
df.dropna()

# Convert data to numpy
X = np.array(df['vit'].tolist())

# 3.2.1 Determining Elbow point using WCSS vs k
wcss_values = []
k_values = []

# Instantiate the model
max_iterations = 10
k_means_model = K_Means_model(k=1, max_iterations=max_iterations)

for k in range(1, 200, 1):
    k_means_model.set_k(k)
    evaluator = Evaluator(k_means_model)
    metrics = evaluator.evaluate(X)
    k_values.append(k)
    wcss_values.append(metrics['cost'])

plt.figure(figsize=(12, 12))
plt.plot(k_values, wcss_values, marker='o')
plt.title(f"k vs WCSS")
plt.xlabel('k: Number of Clusters')
plt.ylabel('WCSS: Within-Cluster sum of squares')
plt.grid(True)
plt.savefig("assignments/2/figures/WCSS_vs_k_1to200.png")

# 3.2.2: K-means clustering with kkmeans1 = 7
k = 7
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)

# 4.2.1
GMM_model = GMM(num_components = 10, max_iterations=10, threshold=1e-6, min_covariance = 1e-6)
GMM_model.fit(X)
means, covariances, weights = GMM_model.get_params()
print(GMM_model.log_likelihood_trace)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(GMM_model.log_likelihood_trace) + 1), GMM_model.log_likelihood_trace, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Plot")
plt.grid(True)
plt.savefig("assignments/2/figures/4_2_1.png")

def calculate_AIC_BIC(X, num_components):
    GMM_model = GMM(num_components=num_components)
    GMM_model.fit(X)
    log_likelihood = GMM_model.log_likelihood_trace[-1]
    # num_params = num_components * (X.shape[1]) * (X.shape[1] + 1) / 2 # Incorrect
    num_params = num_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2) + (num_components - 1)
    return 2 * num_params - 2 * log_likelihood, np.log(X.shape[0]) * num_params - 2 * log_likelihood

AIC_scores = []
BIC_scores = []
num_components_range = range(1, 30)

for num_components in num_components_range:
    print(f"Calculating for num_components: {num_components}")
    AIC, BIC = calculate_AIC_BIC(X, num_components)
    print(f"\tAIC: {AIC}\tBIC: {BIC}")
    AIC_scores.append(AIC)
    BIC_scores.append(BIC)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(num_components_range, AIC_scores, marker='o')
plt.title('AIC Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC Score')

plt.subplot(1, 2, 2)
plt.plot(num_components_range, BIC_scores, marker='o')
plt.title('BIC Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')

plt.tight_layout()
plt.savefig("assignments/2/figures/4_2_2.png")  

# "Lower the AIC and BIC scores are better the clustering"

# Therefore, optimal number of clusters for the 512-dimensional dataset is `kgmm1 = 1`

# GMM on the dataset using the number of clusters as `kgmm1`
GMM_model = GMM(num_components = 1)
X = np.array(df['vit'].tolist())
GMM_model.fit(X)
means, covariances, weights = GMM_model.get_params()
print(GMM_model.log_likelihood_trace)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(GMM_model.log_likelihood_trace) + 1), GMM_model.log_likelihood_trace, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Plot")
plt.grid(True)
plt.savefig("assignments/2/figures/4_2_3.png")

# 4.2.2: Sklearn to support 4.2.1
num_components_range = range(1, 30)
AIC_vals = []
BIC_vals = []
for k in num_components_range:
    sklearn_gmm = GaussianMixture(n_components=k, random_state=64)
    sklearn_gmm.fit(X)
    AIC_vals.append(sklearn_gmm.aic(X))
    BIC_vals.append(sklearn_gmm.bic(X))
    print(f"Calculating for num_components: {k}")
    print(f"\tAIC: {sklearn_gmm.aic(X)}\tBIC: {sklearn_gmm.bic(X)}")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(num_components_range, AIC_vals, marker='o')
plt.title('AIC Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('AIC Score')

plt.subplot(1, 2, 2)
plt.plot(num_components_range, BIC_vals, marker='o')
plt.title('BIC Scores')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC Score')

plt.tight_layout()
plt.savefig("assignments/2/figures/4_2_2_sklearn.png")


# GMM on the dataset using the number of clusters as `kgmm1 = 1`
GMM_model = GMM(num_components = 1)
GMM_model.fit(X)
means, covariances, weights = GMM_model.get_params()
print(GMM_model.log_likelihood_trace)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(GMM_model.log_likelihood_trace) + 1), GMM_model.log_likelihood_trace, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Plot")
plt.grid(True)
plt.savefig("assignments/2/figures/4_2_3.png")

# 5.2.1, 5.2.2, 5.2.3

# Convert into 2D and 3D
X = np.array(df['vit'].tolist())
pca_2d = PCA(num_components=2)
pca_3d = PCA(num_components=3)
pca_2d.fit(X)
print(pca_2d.checkPCA(X))
X_2d = pca_2d.transform(X)
pca_3d.fit(X)
print(pca_3d.checkPCA(X))
X_3d = pca_3d.transform(X)

fig = plt.figure(figsize=(12, 5))

# 2D plot
ax1 = fig.add_subplot(121)
ax1.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
ax1.set_title('2D PCA Visualization')
ax1.set_xlabel(f'First Principal Component\nExplained Variance: {pca_2d.explained_variance_ratio[0]:.2f}')
ax1.set_ylabel(f'Second Principal Component\nExplained Variance: {pca_2d.explained_variance_ratio[1]:.2f}')

# 3D plot
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=X_3d[:, 2], cmap='viridis', alpha=0.7)
ax2.set_title('3D PCA Visualization')
ax2.set_xlabel(f'First PC\nVar: {pca_3d.explained_variance_ratio[0]:.2f}')
ax2.set_ylabel(f'Second PC\nVar: {pca_3d.explained_variance_ratio[1]:.2f}')
ax2.set_zlabel(f'Third PC\nVar: {pca_3d.explained_variance_ratio[2]:.2f}')

# Add a colorbar to the 3D plot
cbar = plt.colorbar(scatter, ax=ax2, pad=0.1)
cbar.set_label('Value of Third Principal Component')

plt.tight_layout()
plt.savefig("assignments/2/figures/5_2_3.png")

# Print the total explained variance for both 2D and 3D
print(f"Total explained variance (2D): {np.sum(pca_2d.explained_variance_ratio):.4f}")
print(f"Total explained variance (3D): {np.sum(pca_3d.explained_variance_ratio):.4f}")

# 6 PCA + Clustering

## 6.1 K-means Clustering Based on 2D Visualization
"""
Performing K-means clustering on the dataset using the number of clusters estimated from the 2D visualization of the dataset `k2 = 5`
"""
k = 5
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)

## 6.2 PCA + K-Means Clustering

"""
Generate a scree plot to identify the optimal number of dimensions for
reduction. Apply dimensionality reduction based on this optimal number
of dimensions. We’ll refer to this as the reduced dataset
"""

num_components = 50
pca = PCA(num_components=num_components)
pca.fit(X)
k_values = range(1, 51)
plt.figure(figsize=(8, 8))
plt.plot(k_values, pca.explained_variance, marker='o')
plt.title("Scree-Plot: eigenvalues (explained_variance) vs num_components")
plt.xlabel("Number of Components")
plt.ylabel("Eigenvalues (explained_variance)")
plt.savefig("assignments/2/figures/6_2_1.png")

# Clearly, we see that the optimal number of dimensions for reduction is `k = 5`.
# `Reduced dataset has 5 components`

"""
Determine the optimal number of clusters for the reduced dataset using
the Elbow Method. We’ll refer to this as kkmeans3 .
"""
pca = PCA(num_components=5)
pca.fit(X)
reduced_dataset = pca.transform(X)

"""
X.shape:  (200, 512)
self.components.shape:  (512, 5)
transformed data shape:  (200, 5)
"""

# Determining Elbow point using WCSS vs k

wcss_values = []
k_values = []

# Instantiate the model
max_iterations = 10
k_means_model = K_Means_model(k=1, max_iterations=max_iterations)

for k in range(1, 50, 1):
    k_means_model.set_k(k)
    evaluator = Evaluator(k_means_model)
    metrics = evaluator.evaluate(reduced_dataset)
    k_values.append(k)
    wcss_values.append(metrics['cost'])

plt.figure(figsize=(12, 12))
plt.plot(k_values, wcss_values, marker='o')
plt.title(f"Elbow-plot: k vs WCSS")
plt.xlabel('k: Number of Clusters')
plt.ylabel('WCSS: Within-Cluster sum of squares')
plt.grid(True)
plt.savefig("assignments/2/figures/6_2_2.png")

"""
Perform K-means clustering on the reduced dataset using number of clus-
ters as kkmeans3=6 .
"""

k = 6
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(reduced_dataset)
evaluator.print_metrics(metrics)

# 6.3 GMM Clustering Based on 2D Visualization

"""
Perform GMMs on the dataset using the number of clusters estimated
from the 2D visualization (k2 ), using the GMM class written by you.
"""

GMM_model = GMM(num_components = 5, threshold=1e-6, min_covariance = 1e-6)
GMM_model.fit(X)
means, covariances, weights = GMM_model.get_params()
print(GMM_model.log_likelihood_trace)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(GMM_model.log_likelihood_trace) + 1), GMM_model.log_likelihood_trace, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Plot")
plt.grid(True)
plt.savefig("assignments/2/figures/6_3_1.png")

"""
Fitting GMM with data shape: (200, 512)
/home/saketh/.local/lib/python3.10/site-packages/scipy/stats/_multivariate.py:583: RuntimeWarning: overflow encountered in exp
  out = np.exp(self._logpdf(x, mean, cov_object))
[137905.3215755012, 137875.90605496283]
"""

## 6.4 PCA + GMM

"""
Determine the optimal number of clusters for the reduced dataset as ob-
tained in 6.2 using AIC or BIC. Let us call this the new kgmm3
"""

def calculate_AIC_BIC(X, num_components):
    GMM_model = GMM(num_components=num_components)
    GMM_model.fit(X)
    silhouette = GMM_model.calculate_silhouette_score(X)
    print(f"Silhoette Score: {silhouette:.4f}")
    log_likelihood = GMM_model.log_likelihood_trace[-1]
    num_params = num_components * (X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2) + (num_components - 1)
    return 2 * num_params - 2 * log_likelihood, np.log(X.shape[0]) * num_params - 2 * log_likelihood

AIC_scores = []
BIC_scores = []
num_components_range = range(1, 200)

for num_components in num_components_range:
    print(f"Calculating for num_components: {num_components}")
    AIC, BIC = calculate_AIC_BIC(reduced_dataset, num_components)
    print(f"\tAIC: {AIC}\tBIC: {BIC}")
    AIC_scores.append(AIC)
    BIC_scores.append(BIC)

min_AIC_idx = np.argmin(AIC_scores)
min_BIC_idx = np.argmin(BIC_scores)

plt.figure(figsize=(12, 10))
plt.plot(num_components_range, AIC_scores, label='AIC', color='blue')
plt.plot(num_components_range, BIC_scores, label='BIC', color='green')

plt.axvline(x=num_components_range[min_AIC_idx], linestyle='--', color='blue', label=f'Lowest AIC at {num_components_range[min_AIC_idx]}')
plt.axvline(x=num_components_range[min_BIC_idx], linestyle='--', color='green', label=f'Lowest BIC at {num_components_range[min_BIC_idx]}')

plt.title('AIC and BIC Scores')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig("assignments/2/figures/6_4_1.png")

"""
Cross-checking with sklearn's GaussianMixture:
"""

num_components_range = range(1, 200)
AIC_vals = []
BIC_vals = []


for k in num_components_range:
    sklearn_gmm = GaussianMixture(n_components=k, random_state=64)
    sklearn_gmm.fit(reduced_dataset)
    AIC_vals.append(sklearn_gmm.aic(reduced_dataset))
    BIC_vals.append(sklearn_gmm.bic(reduced_dataset))
    print(f"Calculating for num_components: {k}")
    print(f"\tAIC: {sklearn_gmm.aic(reduced_dataset)}\tBIC: {sklearn_gmm.bic(reduced_dataset)}")


min_AIC_idx = np.argmin(AIC_vals)
min_BIC_idx = np.argmin(BIC_vals)

plt.figure(figsize=(12, 10))
plt.plot(num_components_range, AIC_vals, label='AIC', color='blue')
plt.plot(num_components_range, BIC_vals, label='BIC', color='green')

plt.axvline(x=num_components_range[min_AIC_idx], linestyle='--', color='blue', label=f'Lowest AIC at {num_components_range[min_AIC_idx]}')
plt.axvline(x=num_components_range[min_BIC_idx], linestyle='--', color='green', label=f'Lowest BIC at {num_components_range[min_BIC_idx]}')

plt.title('AIC and BIC Scores')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig("assignments/2/figures/6_4_1_sklearn.png")

"""
GMM on the dataset using the number of clusters as `kgmm1`
"""
GMM_model = GMM(num_components = 3, threshold=1e-6, min_covariance = 1e-6)
GMM_model.fit(reduced_dataset)
means, covariances, weights = GMM_model.get_params()
print(GMM_model.log_likelihood_trace)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(GMM_model.log_likelihood_trace) + 1), GMM_model.log_likelihood_trace, marker='o')
plt.xlabel("Iterations")
plt.ylabel("Log Likelihood")
plt.title("Log Likelihood Plot")
plt.grid(True)
plt.savefig("assignments/2/figures/6_4_2.png")

# Cluster Analsis

## K-Means Cluster Analysis
"""
`kkmeans1 = 7`; `k2 = 5`; `kkmeans3 = 6`
"""

# For Kkmeans1
k = 7
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)

# For Kkmeans1
k = 7
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(reduced_dataset)
evaluator.print_metrics(metrics)

# For k2
k = 5
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)

# For k2
k = 5
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(reduced_dataset)
evaluator.print_metrics(metrics)

# For Kkmeans3
k = 6
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)

# For Kkmeans3
k = 6
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(reduced_dataset)
evaluator.print_metrics(metrics)

# For Kkmeans3
k = 3
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(reduced_dataset)
evaluator.print_metrics(metrics)


## GMM Cluster Analysis

# For kgmm1 = 1 and dataset = original
gmm = GMM(num_components=1)
gmm.fit(X)
silhouette = gmm.calculate_silhouette_score(X)
db_score = gmm.calculate_davies_bouldin_score(X)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

# For kgmm1 = 1 and dataset = reduced
gmm2 = GMM(num_components=1)
gmm2.fit(reduced_dataset)
silhouette = gmm2.calculate_silhouette_score(reduced_dataset)
db_score = gmm2.calculate_davies_bouldin_score(reduced_dataset)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

# For k2 = 5 and dataset = original
gmm = GMM(num_components=5)
gmm.fit(X)
silhouette = gmm.calculate_silhouette_score(X)
db_score = gmm.calculate_davies_bouldin_score(X)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

# For k2 = 5 and dataset = reduced
gmm2 = GMM(num_components=5)
gmm2.fit(reduced_dataset)
silhouette = gmm2.calculate_silhouette_score(reduced_dataset)
db_score = gmm2.calculate_davies_bouldin_score(reduced_dataset)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

# For kgmm3 = 3 and dataset = original
gmm = GMM(num_components=3)
gmm.fit(X)
silhouette = gmm.calculate_silhouette_score(X)
db_score = gmm.calculate_davies_bouldin_score(X)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

# For kgmm3 = 3 and dataset = reduced
gmm2 = GMM(num_components=3)
gmm2.fit(reduced_dataset)
silhouette = gmm2.calculate_silhouette_score(reduced_dataset)
db_score = gmm2.calculate_davies_bouldin_score(reduced_dataset)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")


# Hierarchical Clustering

class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage_method='average', distance_metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.linkage_matrix = None
        self.labels = None

    def fit(self, X):
        self.linkage_matrix = linkage(X, method=self.linkage_method, metric=self.distance_metric)
        self.labels = self.get_cluster_labels(self.n_clusters)
        return self

    def get_cluster_labels(self, n_clusters):
        return fcluster(self.linkage_matrix, n_clusters, criterion='maxclust') - 1

    def plot_dendrogram(self, figsize=(10, 7), k_values=None):
        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix)
        plt.title(f'Dendrogram (linkage: {self.linkage_method}, metric: {self.distance_metric})')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        if k_values:
            ax = plt.gca()
            for k in k_values:
                cut_distance = self.linkage_matrix[-k+1, 2]
                ax.axhline(y=cut_distance, color='r', linestyle='--', label=f'k={k}')
            plt.legend()
        
        plt.show()

    def compare_with_other_clustering(self, other_labels, other_method_name, k):
        hc_labels = self.get_cluster_labels(k)
        ari = adjusted_rand_score(hc_labels, other_labels)
        ami = adjusted_mutual_info_score(hc_labels, other_labels)
        
        print(f"Comparison with {other_method_name} (k={k}):")
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Adjusted Mutual Information: {ami:.4f}")
        
        print(f"\nHierarchical Clustering (k={k}) cluster sizes:")
        unique, counts = np.unique(hc_labels, return_counts=True)
        for cluster, size in zip(unique, counts):
            print(f"Cluster {cluster + 1}: {size} points")
        print()


class HierarchicalClusteringEvaluator:
    def __init__(self, hc_model: HierarchicalClustering):
        self.hc_model = hc_model

    def evaluate(self, X):
        start_time = time.time()
        self.hc_model.fit(X)
        fit_time = time.time() - start_time

        labels = self.hc_model.labels
        silhouette = self.calculate_silhouette_score(X, labels)
        davies_bouldin = self.calculate_davies_bouldin_score(X, labels)

        return {
            'fit_time': fit_time,
            'num_clusters': self.hc_model.n_clusters,
            'cluster_sizes': self.get_cluster_sizes(labels),
            'silhouette_score': silhouette,
            'davies_bouldin': davies_bouldin,
            'linkage_method': self.hc_model.linkage_method,
            'distance_metric': self.hc_model.distance_metric
        }

    def calculate_silhouette_score(self, X, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(X):
            return silhouette_score(X, labels)
        else:
            return -1  # invalid silhouette score

    def calculate_davies_bouldin_score(self, X, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < len(X):
            return davies_bouldin_score(X, labels)
        else:
            return -1  # invalid davies bouldin score

    def get_cluster_sizes(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def print_metrics(self, metrics):
        print(f"\nHierarchical Clustering Results:")
        print(f"Number of Clusters: {metrics['num_clusters']}")
        print(f"Linkage Method: {metrics['linkage_method']}")
        print(f"Distance Metric: {metrics['distance_metric']}")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"Davies Bouldin Score: {metrics['davies_bouldin']:.4f}")
        print(f"Fit time: {metrics['fit_time']:.4f} seconds")
        print(f"Cluster sizes:")
        for cluster, size in metrics['cluster_sizes'].items():
            print(f"\tCluster {cluster + 1}: {size} points")

def experiment_hierarchical_clustering(X):
    linkage_methods = ['single', 'complete', 'average', 'centroid', 'median']
    distance_metrics = ['euclidean', 'cosine']

    for linkage_method in linkage_methods:
        for distance_metric in distance_metrics:
            if (linkage_method == 'centroid' or linkage_method == 'median') and distance_metric != 'euclidean':
                continue  # Ward linkage only works with Euclidean distance
            hc = HierarchicalClustering(n_clusters=3, linkage_method=linkage_method, distance_metric=distance_metric)
            evaluator = HierarchicalClusteringEvaluator(hc)
            
            metrics = evaluator.evaluate(X)
            evaluator.print_metrics(metrics)
            
            hc.plot_dendrogram()

def compare(X, kmeans_labels, gmm_labels):
    hc = HierarchicalClustering(n_clusters=3, linkage_method='complete', distance_metric='euclidean')
    evaluator = HierarchicalClusteringEvaluator(hc)
    
    # Fit the model and print general metrics
    metrics = evaluator.evaluate(X)
    # evaluator.print_metrics(metrics)
    
    # Compare with K-means (k=5) and GMM (k=3)
    hc.compare_with_other_clustering(kmeans_labels, "K-means", 5)
    hc.compare_with_other_clustering(gmm_labels, "GMM", 3)
    
    # Plot dendrogram with cut lines for k=3 and k=5
    hc.plot_dendrogram(k_values=[3, 5])
    

# To plot dendograms for all combinations, call the function
experiment_hierarchical_clustering(X)

# Comparing Kmeans and GMM clusters with Hierarchial Clustering

# For k2
k = 5
n_iterations = 100
kmeans = K_Means_model(k, max_iterations=n_iterations, plot_fitting_graph=True)
evaluator = Evaluator(kmeans)
metrics = evaluator.evaluate(X)
evaluator.print_metrics(metrics)
kmeans_labels = kmeans.predict(X)

# kgmm3 = 3
gmm = GMM(num_components=3)
gmm.fit(X)
silhouette = gmm.calculate_silhouette_score(X)
db_score = gmm.calculate_davies_bouldin_score(X)
gmm_labels = gmm.predict(X)
print(f"Silhoette Score: {silhouette:.4f}")
print(f"Davies Bouldin Score: {db_score:.4f}")

compare(X, kmeans_labels, gmm_labels)

"""
Results:

Comparison with K-means (k=5):
Adjusted Rand Index: 0.1914
Adjusted Mutual Information: 0.2564

Hierarchical Clustering (k=5) cluster sizes:
Cluster 1: 7 points
Cluster 2: 68 points
Cluster 3: 120 points
Cluster 4: 4 points
Cluster 5: 1 points

Comparison with GMM (k=3):
Adjusted Rand Index: 0.0126
Adjusted Mutual Information: 0.0261

Hierarchical Clustering (k=3) cluster sizes:
Cluster 1: 75 points
Cluster 2: 120 points
Cluster 3: 5 points
"""


# Nearest Neighbour Search

## PCA + KNN

# Read the data
df = pd.read_csv("data/external/spotify.csv")
df.dropna()
features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
# normalized_df['track_genre'] = df['track_genre'] # don't add
normalized_df.head()

# Create a new column 'vit' which is a list of all other column values for each row
normalized_df['vit'] = normalized_df.apply(lambda row: row.tolist(), axis=1)
normalized_df['vit'].shape
X = np.array(normalized_df['vit'].tolist())

"""
Scree Plot
"""
num_components = 10
pca = PCA(num_components=num_components)
pca.fit(X)
k_values = range(1, 11)
plt.figure(figsize=(8, 8))
plt.plot(k_values, pca.explained_variance, marker='o')
plt.title("Scree-Plot: eigenvalues (explained_variance) vs num_components")
plt.xlabel("Number of Components")
plt.ylabel("Eigenvalues (explained_variance)")
plt.savefig("assignments/2/figures/9_1_1.png")

"""
Optimal Number of dimensions would be `3`
"""
"""
Use the KNN model implemented in Assignment 1 on the reduced dataset
using the best {k, distance metric} pair obtained.
"""

pca = PCA(num_components=3)
pca.fit(X)
reduced_dataset = pca.transform(X)

## Evaluation

pca_df = pd.DataFrame(reduced_dataset, columns=['col1', 'col2', 'col3'])
pca_df['track_genre'] = df['track_genre']

features_list = ['col1', 'col2', 'col3']
k = 15
metric = 'cosine'
model = Best_KNN_model(k, metric, features=features_list)
model.train(pca_df)
model.split_data(validation_split=0.1, test_split=0.1)
evaluator = Best_KNN_evaluate(model)
validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
evaluator.print_metrics(validation_metrics, "Validation")
acc = validation_metrics['accuracy']
macro_p = validation_metrics['macro_p']
macro_r = validation_metrics['macro_r']
macro_f1 = validation_metrics['macro_f1']
micro_p = validation_metrics['micro_p']
micro_r = validation_metrics['micro_r']
micro_f1 = validation_metrics['micro_f1']
results = []
results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))
sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
df_print = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df_print.to_string(index=False))

"""
Predictions made: 1000/11400
Predictions made: 2000/11400
Predictions made: 3000/11400
Predictions made: 4000/11400
Predictions made: 5000/11400
Predictions made: 6000/11400
Predictions made: 7000/11400
Predictions made: 8000/11400
Predictions made: 9000/11400
Predictions made: 10000/11400
Predictions made: 11000/11400

Validation Set Results:
accuracy: 0.0875
macro_p: 0.0838
macro_r: 0.0882
macro_f1: 0.0810
micro_p: 0.0875
micro_r: 0.0875
micro_f1: 0.0875
avg_time: 0.0163
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric
0.087544 0.083822 0.088198  0.080998 0.087544 0.087544  0.087544 15 cosine
"""

"""
Plot the inference time for the KNN model on both the complete dataset
and the reduced dataset. Comment on the differences and implications of
the inference times.
"""

# Example objects
models = [Best_KNN_model]
datasets = [df, pca_df]
evaluators = [Best_KNN_evaluate]
dataset_names = ['original', 'reduced']
features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
features_list = ['col1', 'col2', 'col3']

# Sizes of datasets to test
n_rows = [1000, 5000, 10000, 20000, 40000, 60000, 100000]

# Initialize dictionary to store inference times with dataset names as keys
inference_times = {name: [] for name in dataset_names}

# Iterate through each dataset
for dataset, dataset_name in zip(datasets, dataset_names):
    # print(dataset.info())
    for n_row in n_rows:
        print(f"Processing {dataset_name} dataset with {n_row} rows")
        
        # Start timing
        start_time = time.time()
        
        # Create and train the model
        if dataset_name == "original":
            obj = Best_KNN_model(15, 'cosine', features_to_normalize)
        else:
            obj = Best_KNN_model(15, 'cosine', features_list)
        obj.train(dataset[:n_row])  # Train with the selected number of rows
        
        # Split data for validation and testing
        obj.split_data(validation_split=0.1, test_split=0.1)
        
        # Evaluate the model
        evaluator = Best_KNN_evaluate(obj)
        validation_metrics = evaluator.evaluate(obj.X_valid, obj.y_valid)
        
        # End timing
        end_time = time.time()
        
        # Store inference time
        inference_times[dataset_name].append(end_time - start_time)

# Plotting inference times
for dataset_name, times in inference_times.items():
    plt.plot(n_rows, times, label=dataset_name)

plt.xlabel('Training Dataset Size')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Training Dataset Size')
plt.legend()
plt.savefig("assignments/2/figures/9_2_3.png")

