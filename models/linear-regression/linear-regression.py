# Import Necessary libraries
import numpy as np # Matrix calculations
import pandas as pd # reading data
import matplotlib.pyplot as plt # plotting graphs
import matplotlib.gridspec as gridspec

class LinearRegression:
    def __init__(self, learning_rate = 0.01, num_steps = 1000, lambda_regularization = 0):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.lambda_regularization = lambda_regularization # lambda
        self.weights = None # beta1
        self.bias = None # beta0

    def split_data(self, data, validation_split=0.1, test_split=0.1):
        shuffled_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(validation_split * len(data))
        test_size = int(test_split * len(data))
        train_size = len(data) - (valid_size + test_size)

        X = data['x'].values.reshape(-1, 1)
        y = data['y'].values.reshape(-1, 1)

        self.X_train = shuffled_df.iloc[:train_size, :-1].values
        self.y_train = shuffled_df.iloc[:train_size, -1].values
        self.X_val = shuffled_df.iloc[train_size:train_size + valid_size, :-1].values
        self.y_val = shuffled_df.iloc[train_size:train_size + valid_size, -1].values
        self.X_test = shuffled_df.iloc[train_size + valid_size:, :-1].values
        self.y_test = shuffled_df.iloc[train_size + valid_size:, -1].values

    def train(self, X, y):
        self.fit(X, y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_steps):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Gradient descent
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + self.lambda_regularization * self.weights)
            db = (1 / n_samples) * (np.sum(y_predicted - y))

            # backpropagate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    

class PolynomialRegression:
    def __init__(self, k=2, learning_rate=0.01, num_steps=1000, regularization = 'none', lambda_regularization=0):
        self.k = k # degree
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.regularization = regularization.lower()
        self.lambda_regularization = lambda_regularization
        self.weights = None
        self.bias = None

    def set_k(self, k):
        self.k = k

    def polynomial_features(self, X):
        poly_features = X
        for i in range(2, self.k + 1):
            poly_features = np.concatenate((poly_features, X ** i), axis=1)
        return poly_features

    def split_data(self, data, validation_split=0.1, test_split=0.1):
        shuffled_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(validation_split * len(data))
        test_size = int(test_split * len(data))
        train_size = len(data) - (valid_size + test_size)

        self.X_train = shuffled_df.iloc[:train_size, :-1].values
        self.y_train = shuffled_df.iloc[:train_size, -1].values
        self.X_val = shuffled_df.iloc[train_size:train_size + valid_size, :-1].values
        self.y_val = shuffled_df.iloc[train_size:train_size + valid_size, -1].values
        self.X_test = shuffled_df.iloc[train_size + valid_size:, :-1].values
        self.y_test = shuffled_df.iloc[train_size + valid_size:, -1].values

    def train(self, X, y, save_frames = False, path_to_image_folder = ".", save_steps=10, seed = None):
        self.fit(X, y, save_frames, path_to_image_folder, save_steps, seed)

    def fit(self, X, y, save_frames = False, path_to_image_folder = ".", save_steps = 10, seed = None):
        X_poly = self.polynomial_features(X)
        n_samples, n_features = X_poly.shape
        if seed == None:
            self.weights = np.zeros(n_features)
        else:
            np.random.seed(seed)
            self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        for i in range(self.num_steps):
            y_predicted = np.dot(X_poly, self.weights) + self.bias

            if self.regularization == 'none':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)))
            elif self.regularization == 'l1':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)) + self.lambda_regularization * np.sign(self.weights))
            elif self.regularization == 'l2':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)) + self.lambda_regularization * self.weights)
            else:
                raise ValueError("Invalid regularization type. Should be none or l1 or l2 according to doc.")
            
            db = (1 / n_samples) * (np.sum(y_predicted - y))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if save_frames == True:
                if i % save_steps == 0 or i == self.num_steps - 1:  # Save image every 10th step
                    self.plot_convergence(i, y, y_predicted, path_to_image_folder)

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
    
    def plot_convergence(self, step, y_true, y_pred, path_to_image_folder = "."):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

        ax0 = plt.subplot(gs[0])
        ax0.scatter(self.X_train, self.y_train, color='b', alpha=0.3, label='Training data')
        sorted_indices = np.argsort(self.X_train[:, 0])
        X_sorted = self.X_train[sorted_indices]
        y_sorted_pred = y_pred[sorted_indices]
        ax0.plot(X_sorted, y_sorted_pred, color='black', label='Fitted curve')
        ax0.set_title(f'Fitted Curve at Step {step}')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.legend()
        ax0.grid(True)

        ax1 = plt.subplot(gs[1])
        mse = np.mean((y_true - y_pred) ** 2)
        ax1.bar(['MSE'], [mse], color='orange')
        ax1.set_title('MSE')
        ax1.set_ylim([0, max(1, mse)])

        ax2 = plt.subplot(gs[2])
        variance = np.var(y_pred)
        ax2.bar(['Variance'], [variance], color='green')
        ax2.set_title('Variance')
        ax2.set_ylim([0, max(1, variance)])

        ax3 = plt.subplot(gs[3])
        std_dev = np.std(y_pred)
        ax3.bar(['Standard Deviation'], [std_dev], color='red')
        ax3.set_title('Standard Deviation')
        ax3.set_ylim([0, max(1, std_dev)])

        plt.tight_layout()

        # Save the frame
        filename = f"{path_to_image_folder}frame_{step}.png"
        plt.savefig(filename)
        plt.close()
    
    def save_model(self, filename):
        np.savez(filename, weights=self.weights, bias=self.bias)

    def load_model(self, filename):
        model_data = np.load(filename)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
    

