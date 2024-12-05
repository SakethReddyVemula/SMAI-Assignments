# Import necessary libraries
# Import necessary libraries
import pandas as pd
import numpy as np
from typing import List, Tuple
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.special import logsumexp

class GMM:
    def __init__(self, num_components: int, max_iterations: int=100, threshold: float=1e-6, min_covariance: float=1e-6, seed: int=42):
        self.num_components = num_components # K
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.min_covariance = min_covariance # Ensures regularization
        self.seed = seed # Random initialization
        self.means = None # Mu_k/mean_k
        self.covariances = None # Sigma_k
        self.weights = None # Phi_k (prior)
        self.log_likelihood = None
        self.log_likelihood_trace = []

    def initialize_GMM_parameters(self, X: np.ndarray)-> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        num_samples, num_features = X.shape # (200, 512)
        # print(f"num_samples: {num_samples}, num_features: {num_features}")
        np.random.seed(self.seed)

        # Initialize means
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        means = np.random.uniform(min_vals, max_vals, (self.num_components, num_features))

        # Initialize covariances
        covariances = []
        for _ in range(self.num_components):
            A = np.random.randn(num_features, num_features)
            cov = np.dot(A.T, A) + self.min_covariance * np.eye(num_features) # Add term self.min_covariance * np.eye(num_features) to diagonal of initial covariances to ensure they are POSITIVE SEMI-DEFINITE
            covariances.append(cov)

        # Initialize weights
        weights = np.random.rand(self.num_components)
        weights /= weights.sum()
        # print(means.shape)
        # print(len(covariances))
        # print(covariances[0].shape)
        # print(weights.shape)
        return means, covariances, weights # (means->(3, 512), covariances->List of (512, 512) of length 3, weights->(3,))
    
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        print(f"Fitting GMM with data shape: {X.shape}")

        if X.shape[0] == 0:
            raise ValueError("All data points contain NaN or Inf values. Cannot fit the model.")
        
        # Initialize parameters using initialize_GMM_parameters function
        self.means, self.covariances, self.weights = self.initialize_GMM_parameters(X)
        # print("Iterations: ", self.max_iterations)
        
        for iter in range(self.max_iterations):
            # print(f"iteration: {iter}")
            # E-step (Expectation)
            """
                Compute the posterior probability over Z given our current
                model. i.e., how much do we think each Guassian generates each
                datapoint
            """
            responsibilities = self.E_step(X)

            # M-step (Maximization)
            """
                Update the parameters of each gaussian to maximize the probability so
                that it maximizes the prior. Assuming that the 1st step correct
            """
            self.M_step(X, responsibilities)

            # log-likelihood calculation
            new_log_likelihood = self.get_likelihood(X)
            
            # check for convergence
            if self.log_likelihood is not None:
                if np.isnan(new_log_likelihood) or np.isinf(new_log_likelihood):
                    print(f"Warning: Invalid log-likelihood. Stopping Early.")
                    break

                if np.abs(new_log_likelihood - self.log_likelihood) < self.threshold:
                    break

            self.log_likelihood = new_log_likelihood
            # print(new_log_likelihood)
            # Add to trace
            self.log_likelihood_trace.append(new_log_likelihood)
        
        # print(len(self.log_likelihood_trace))

        return self.means, self.covariances, self.weights

    def E_step(self, X: np.ndarray) -> np.ndarray:
        EPSILON = 1e-10
        # getting log probs first, and then the actual exp probs.
        log_probabilities = np.array([np.log(self.weights[k]) + multivariate_normal(mean=self.means[k], cov=self.covariances[k]).logpdf(X) for k in range(self.num_components)]).T
        log_probabilities_max = np.max(log_probabilities, axis=1, keepdims=True)
        exp_probabilities = np.exp(log_probabilities - log_probabilities_max)
        return exp_probabilities / (np.sum(exp_probabilities, axis=1, keepdims=True) + EPSILON)
    
    def M_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
            In code: gamme == responsibilities
            N_k = summation(n=1toK) gamma_k(n)
            mean_k = (1/N_k) summation(n=1toK) (gamma_k(n).x(n))
            covariance_k = (1 / N_k) summation(n=1toK) ((gamma_k(n) (x(n) - mean_k) (x(n) - mean_k).T))
            weights_k (or Phi_k) = N_k / N
        """

        N = responsibilities.sum(axis = 0) # N_k = summation(n=1toK) gamma_k(n)
        N = np.maximum(N, 1e-8) # To Avoid division by zero
        
        # Update means
        self.means = np.dot(responsibilities.T, X) / N[:, np.newaxis]
        
        # Update covariances
        for k in range(self.num_components):
            diff = X - self.means[k] # x(n) - mean_k
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N[k] + self.min_covariance * np.eye(X.shape[1])
        
        # Update weights (Phi_k)
        self.weights = N / X.shape[0] # N_k / N (where N -> num_samples)
        
    def get_params(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        return self.means, self.covariances, self.weights
    
    def get_membership(self, X: np.ndarray) -> np.ndarray:
        return self.E_step(X)
    
    def get_likelihood(self, X: np.ndarray) -> float:
        """
            ln(P(X|Phi, mu, Sigma)) = sum(n=1toN) ln( sum(k=1toK) (Phi_k) (N (x(n) | mu_k, sigma_k)))
        """

        likelihood = 0
        
        for k in range(self.num_components):
            pdf_values = self.pdf(X, self.means[k], self.covariances[k])
            pdf_values = np.nan_to_num(pdf_values, nan=1e-300, posinf=1e300, neginf=1e-300)
            likelihood += self.weights[k] * pdf_values
        
        likelihood = np.maximum(likelihood, 1e-300)  # Avoid log(0)
        log_likelihood = np.sum(np.log(likelihood))
        return log_likelihood if np.isfinite(log_likelihood) else -np.inf

    
    def pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        try:
            return multivariate_normal.pdf(X, mean, cov)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular covariance matrix. Using diagonal covariance.")
            diagonal_cov = np.diag(np.diag(cov)) + self.min_covariance * np.eye(cov.shape[0])
            return multivariate_normal.pdf(X, mean, diagonal_cov)
        
    def calculate_silhouette_score(self, X: np.ndarray) -> float:
        """
        Calculate the silhouette score for the GMM clustering.
        """
        # Predict cluster labels
        labels = self.predict(X)
        
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        print(f"Number of unique labels: {n_labels}")
        print(f"Unique labels: {unique_labels}")
        print(f"Label counts: {[np.sum(labels == i) for i in unique_labels]}")
        
        if n_labels == 1:
            print("Warning: All samples were assigned to the same cluster.")
            return -1

        # Calculate silhouette score
        try:
            return silhouette_score(X, labels)
        except ValueError as e:
            print(f"Error calculating silhouette score: {e}")
            return -1
        
    def calculate_davies_bouldin_score(self, X: np.ndarray) -> float:
        """
            Calculate the davies boulding score for the GMM Clustering.
        """
        # Predict cluster labels
        labels = self.predict(X)

        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        print(f"Number of Unique labels: {n_labels}")
        print(f"Unique labels: {unique_labels}")
        print(f"Label counts: {[np.sum(labels == i) for i in unique_labels]}")

        if n_labels == 1:
            print("Warning: All samples were assigned to the same cluster.")
            return -1
        
        # Calculate silhouette score
        try:
            return davies_bouldin_score(X, labels)
        except ValueError as e:
            print(f"Error calculating davies bouldin score: {e}")
            return -1

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely cluster for each data point.
        """
        responsibilities = self.E_step(X)
        predictions = np.argmax(responsibilities, axis=1)
        
        print("Prediction statistics:")
        print(f"Unique predictions: {np.unique(predictions)}")
        print(f"Prediction counts: {np.bincount(predictions)}")
        
        return predictions