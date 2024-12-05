# Import necessary libraries
import numpy as np

class PCA:
    def __init__(self, num_components):
        self.num_components = num_components # Estimated
        self.components = None # np.ndarray of shape (num_components, num_features)
        self.mean = None # np.ndarray of shape (num_features,)
        self.explained_variance_ratio = None

    def get_num_components(self):
        return self.num_components
    
    def set_num_components(self, num_components):
        self.num_components = num_components
    
    def fit(self, X): # X.shape = (200, 512)
        # STEP-1: center the data about its mean
        self.mean = np.mean(X, axis=0) # mu
        X_centered = X - self.mean

        # STEP-2: compute the covariance matrix (sigma)
        sigma = np.cov(X_centered, rowvar=False) # sigma is covariance matrix

        # STEP-3: compute eigenvectors vi's and eigenvalues (lambda's)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first num_components eigenvectors (PCA step)
        self.components = eigenvectors[:, :self.num_components]

        # calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance = eigenvalues[:self.num_components]
        self.explained_variance_ratio = eigenvalues[:self.num_components] / total_variance

    def transform(self, X):
        # center the data
        X_centered = X - self.mean

        # project the data onto the principal components
        transformed = np.dot(X_centered, self.components)
        print("X.shape: ", X.shape)
        print("self.components.shape: ", self.components.shape)
        print("transformed data shape: ", transformed.shape)
        return transformed
    
    def checkPCA(self, X):
        self.fit(X)
        X_transformed = self.transform(X)

        # Check if the shape is correct
        shape_check = X_transformed.shape == (X.shape[0], self.num_components)
        
        # Check if the variance is preserved (approximately)
        original_var = np.var(X, axis=0).sum()
        transformed_var = np.var(X_transformed, axis=0).sum()
        variance_check = np.isclose(transformed_var, np.sum(self.explained_variance_ratio * original_var), rtol=1e-2)
        
        print(f"Shape check: {shape_check}")
        print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")
        print(f"Variance check: {variance_check}")
        print(f"Original variance: {original_var}, Transformed variance: {transformed_var}")
        print(f"Explained variance ratio: {self.explained_variance_ratio}")
        print(f"Sum of explained variance ratio: {np.sum(self.explained_variance_ratio)}")
        
        return shape_check and variance_check


