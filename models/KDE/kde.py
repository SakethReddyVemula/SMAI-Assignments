import numpy as np
import matplotlib.pyplot as plt

class KDE:
    def __init__(
        self,
        kernel:str="gaussian",
        bandwidth:float=1.0
    ):
        self.kernel = kernel.lower()
        self.bandwidth = bandwidth # parameter h: aka smoothing parameter of bumps
        self.data = None
        self.n_samples = 0
        self.n_features = 0

    def _gaussian_kernel(self, x):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    
    def _box_kernel(self, x):
        return np.where(np.abs(x) <= 1, 0.5, 0)
    
    def _triangular_kernel(self, x):
        return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
    
    def _get_kernel(self, x):
        if self.kernel == 'gaussian':
            return self._gaussian_kernel(x)
        elif self.kernel == 'box':
            return self._box_kernel(x)
        elif self.kernel == 'triangular':
            return self._triangular_kernel(x)
        else:
            raise ValueError("Unsupported kernel type")
    
    def fit(self, X):
        self.data = np.array(X)
        self.n_samples, self.n_features = self.data.shape
        return self
    
    def predict(self, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        densities = np.zeros(len(X))

        for i in range(len(X)):
            distances = np.linalg.norm(self.data - X[i], axis=1) / self.bandwidth
            densities[i] = np.mean(self._get_kernel(distances)) / self.bandwidth

        return densities
    
    def visualize(self, X, y=None):
        """
        Visualize the density estimate for 2D data
        
        Parameters:
        X (array-like): 2D data points
        y (array-like, optional): Labels for coloring points
        """
        if self.n_features != 2:
            raise ValueError("Visualization only supported for 2D data")
            
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        Z = self.predict(positions).reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, levels=20, cmap='viridis')
        plt.colorbar(label='Density')
        plt.scatter(X[:, 0], X[:, 1], c='white', alpha=0.5, s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('KDE Density Estimation')
        plt.grid(True)
        plt.savefig(f"figures/kde_kernel_{self.kernel}_bandwidth_{self.bandwidth}.png")
        plt.show()
        

    
    