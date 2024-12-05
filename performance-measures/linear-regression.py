# Import Necessary libraries
import numpy as np # Matrix calculations
import pandas as pd # reading data
import matplotlib.pyplot as plt # plotting graphs
import matplotlib.gridspec as gridspec


class LR_EvaluationMetrics:
    def __init__(self, model):
        self.model = model

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def variance(self, y):
        return np.var(y)
    
    def std_dev(self, y):
        return np.std(y)
    
    def calculate_metrics(self):
        y_train_pred = self.model.predict(self.model.X_train)
        y_val_pred = self.model.predict(self.model.X_val)
        y_test_pred = self.model.predict(self.model.X_test)

        train_mse = self.mse(self.model.y_train, y_train_pred)
        valid_mse = self.mse(self.model.y_val, y_val_pred)
        test_mse = self.mse(self.model.y_test, y_test_pred)
        train_var = self.variance(y_train_pred)
        valid_var = self.variance(y_val_pred)
        test_var = self.variance(y_test_pred)
        train_sd = self.std_dev(y_train_pred)
        valid_sd = self.std_dev(y_val_pred)
        test_sd = self.std_dev(y_test_pred)

        return (train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd)
    
    def plot_without_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        plt.title('Training data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        y_pred = self.model.predict(self.model.X_train)

        plt.plot(self.model.X_train, y_pred, color='r', label='Fitted line')

        plt.title('Training data with Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        y_pred = self.model.predict(self.model.X_train)

        plt.plot(self.model.X_train, y_pred, color='black', label='Fitted line')

        plt.title('Training data with Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

class PR_EvaluationMetrics:
    def __init__(self, model):
        self.model = model 

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def variance(self, y):
        return np.var(y)

    def std_dev(self, y):
        return np.std(y)

    def calculate_metrics(self):
        y_train_pred = self.model.predict(self.model.X_train)
        y_val_pred = self.model.predict(self.model.X_val)
        y_test_pred = self.model.predict(self.model.X_test)

        train_mse = self.mse(self.model.y_train, y_train_pred)
        valid_mse = self.mse(self.model.y_val, y_val_pred)
        test_mse = self.mse(self.model.y_test, y_test_pred)
        train_var = self.variance(y_train_pred)
        valid_var = self.variance(y_val_pred)
        test_var = self.variance(y_test_pred)
        train_sd = self.std_dev(y_train_pred)
        valid_sd = self.std_dev(y_val_pred)
        test_sd = self.std_dev(y_test_pred)

        return (train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd)
    
    def plot_without_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        plt.title('Training data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        # Sorting the training data to get a smooth curve
        sorted_indices = np.argsort(self.X_train[:, 0])
        X_sorted = self.model.X_train[sorted_indices]
        
        # Generating predictions for the sorted X values
        y_pred = self.model.predict(X_sorted)
        
        # Plotting the fitted polynomial curve
        plt.plot(X_sorted, y_pred, color='black', label='Fitted curve')

        plt.title('Training, Validation, and Test Data with Fitted Polynomial Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph_wo_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        plt.title('Training, Validation, and Test Data with Fitted Polynomial Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph_w_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        # Sorting the training data to get a smooth curve
        sorted_indices = np.argsort(self.model.X_train[:, 0])
        X_sorted = self.model.X_train[sorted_indices]
        
        # Generating predictions for the sorted X values
        y_pred = self.model.predict(X_sorted)
        
        # Plotting the fitted polynomial curve
        plt.plot(X_sorted, y_pred, color='black', label='Fitted curve')

        plt.title(f'Training, Validation, and Test Data with Fitted Polynomial Curve (k = {self.model.k})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../../assignments/1/figures/{self.model.regularization}_{self.model.k}.png")
        # plt.show()

