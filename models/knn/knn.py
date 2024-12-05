# Imports necessary for K-NN
import pandas as pd
import numpy as np
from numpy.linalg import norm # Uses: cosine similarity, ...
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from numpy import linalg


"""
    n: Number of training samples (rows in the dataset)
    m: Number of features used for distance computatio)
    k: Number of nearest neighbors considered (k-value)
    f: Number of samples to predict (size of the test set)
    d: Distance metric used (manhattan, euclidean, cosine)
    L: Number of unique labels for storing the count of each label
"""
class KNN_model:
    # TC: O(1); SC: O(m) -> to store the list of features and other attributes
    def __init__(self, k: int, distance_metrics: str, features: list):
        # Initialize the KNN model with the number of neighbors (k),
        # the distance metric to use ('manhattan', 'euclidean', or 'cosine'),
        # and the list of features to consider for distance calculation.
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_data = None # Placeholder for the training dataset
        self.prediction_count = 0 # Counter for the number of predictions made
        self.total_time_taken = 0 # Accumulator for total time taken to make predictions

    # TC: O(1); SC: O(1) 
    def get_k(self):
        # Getter method for k (number of neighbors)
        return self.k
    
    # TC: O(1); SC: O(1)
    def set_k(self, k):
        # Setter method for k (allows updating the number of neighbors)
        self.k = k

    # TC: O(1); SC: O(1)
    def get_distance_metrics(self):
        # Getter method for the distance metric
        return self.distance_metrics
    
    # TC: O(1); SC: O(1)
    def set_distance_metrics(self, distance_metrics):
        # Setter method for the distance metric (allows updating the metric)
        self.distance_metrics = distance_metrics

    # TC: O(m); SC: O(1)
    def get_distances(self, row1, row2):
        # Compute the distance between two rows based on the selected distance metric
        if self.distance_metrics == 'manhattan':
            # Manhattan distance (sum of absolute differences)
            distance = 0.0
            for feature in self.features:
                diff = abs(row1[feature] - row2[feature])
                distance += diff
            return distance
        elif self.distance_metrics == 'euclidean':
            # Euclidean distance (sum of squared differences)
            distance = 0.0
            for feature in self.features:
                diff = row1[feature] - row2[feature]
                distance += diff
            return distance
        elif self.distance_metrics == 'cosine':
            # Cosine distance (1 - cosine similarity)
            dot_product = 0.0
            norm1 = 0.0
            norm2 = 0.0
            for feature in self.features:
                dot_product += row1[feature] * row2[feature]
                norm1 += row1[feature] * row1[feature]
                norm2 += row2[feature] * row2[feature]
            if norm1 == 0 and norm2 == 0:
                return 1.0 # Handle the case where both norms are zero
            cosine_similarity = dot_product / ((norm1 ** 0.5) * (norm2 ** 0.5))
            return (1.0 - cosine_similarity)  # Return cosine distance
        else:
            # Raise an error if an invalid metric is provided
            return ValueError("invalid metric")

    # TC: O(1); SC: O(nm)
    def train(self, df):
        # Store the training data
        self.train_data = df

    # TC: O(n); SC: O(n)
    def split_data(self, validation_split=0.1, test_split=0.1):
        # Split the dataset into training, validation, and test sets
        self.validation_split = validation_split
        self.test_split = test_split

        # Shuffle the dataset and reset indices
        shuffled_df = self.train_data.sample(frac = 1).reset_index(drop=True) # Shuffle total df with dropping na
        
        # Calculate the sizes of the validation and test sets
        self.validation_size = int(len(shuffled_df) * validation_split)
        self.test_size = int(len(shuffled_df) * test_split)
        print(f"validation size: {self.validation_size}\ttest_size: {self.test_size}\n")
        
        # Extract the test, validation, and training sets from the shuffled data
        self.test_set = shuffled_df.iloc[: self.test_size]
        self.valid_set = shuffled_df.iloc[self.test_size : self.test_size + self.validation_size]
        self.train_set = shuffled_df.iloc[self.test_size + self.validation_size : ]

    # TC: O(k); SC: O(L)
    def get_majority(self, nearest_neighbours):
        # Determine the majority label among the nearest neighbors
        label_counts = {}

        # Count the occurrences of each label in the nearest neighbors
        for i, _ in nearest_neighbours:
            label = self.train_data.iloc[i]['track_genre']  # Use the index to get the label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Return the label with the highest count (majority vote)
        predicted_label = max(label_counts, key=label_counts.get)
        return predicted_label

    # TC: (nm + nlogn + k) (computing distances + sorting the distances + determining majority label)
    # SC: O(n) -> store the distances for all 'n' training samples
    def predict_a_sample(self, test_row):
        # Predict the label for a single test sample
        start_time = time.time()
        distances = []

        # Calculate distances between the test sample and all training samples
        for i, train_row in self.train_data.iterrows():
            distance = self.get_distances(test_row, train_row)
            distances.append((i, distance))

        # Sort distances and select the k-nearest neighbors
        distances.sort(key=lambda x: x[1])
        nearest_neighbours = distances[:self.k]

        # Get the majority label from the k-nearest neighbors
        prediction = self.get_majority(nearest_neighbours)

        # Calculate time taken for prediction and update totals
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1

        # Optionally print progress every 10 predictions
        if self.prediction_count % 10 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(f (nm + n log n + k)) -> make prediction for each of the 'f' test samples
    # SC: O(f n) -> same
    def predict(self, X_test):
        # Predict labels for all samples in the test set
        self.prediction_count = 0 # Reset prediction counter 
        self.total_time_taken = 0 # Reset time counter

        # Generate predictions for each sample in the test set
        predictions = [self.predict_a_sample(row) for _, row in X_test.iterrows()]
        return predictions

class Vectorized_KNN_model:
    # TC: O(1) - Simply assigns values to variables.
    # SC: O(d) - Space used to store the features list.
    def __init__(self, k: int, distance_metrics: str, features: list):
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_embeddings = None  # Numpy array for training embeddings
        self.train_labels = None  # Numpy array for training labels
        self.prediction_count = 0
        self.total_time_taken = 0

    # TC: O(1) - Simple getter function
    def get_k(self):
        return self.k
    
    # TC: O(1) - Simple setter function.
    def set_k(self, k):
        self.k = k

    # TC: O(1) - Simple getter function.
    def get_distance_metrics(self):
        return self.distance_metrics
    
    # TC: O(1) - Simple setter function.
    def set_distance_metrics(self, distance_metrics):
        self.distance_metrics = distance_metrics

    # TC:
    # - Manhattan: O(n * d) - Computes absolute differences and sums them across all features.
    # - Euclidean: O(n * d) - Computes squared differences, sums them, and takes the square root.
    # - Cosine: O(n * d) - Computes dot product and norms, then performs division.
    # SC: O(n) - Space to store the distances for each training sample.
    def calculate_distances(self, test_embedding):
        if self.distance_metrics == 'manhattan':
            distances = np.sum(np.abs(self.train_embeddings - test_embedding), axis=1)
        elif self.distance_metrics == 'euclidean':
            distances = np.sqrt(np.sum((self.train_embeddings - test_embedding) ** 2, axis=1))
        elif self.distance_metrics == 'cosine':
            dot_product = np.dot(self.train_embeddings, test_embedding)
            norms = np.linalg.norm(self.train_embeddings, axis=1) * np.linalg.norm(test_embedding)
            distances = 1 - (dot_product / norms)
        else:
            raise ValueError("Invalid distance metric")
        return distances
    
    # TC: O(n) - Shuffling and splitting data.
    # SC: O(n * d) - Space to store embeddings and labels after splitting.
    def split_data(self, validation_split=0.1, test_split=0.1):
        self.validation_split = validation_split
        self.test_split = test_split
        shuffled_df = self.train_data.sample(frac = 1).reset_index(drop=True) # shuffle total df with dropping na
        self.validation_size = int(len(shuffled_df) * validation_split)
        self.test_size = int(len(shuffled_df) * test_split)
        print(f"validation size: {self.validation_size}\ttest_size: {self.test_size}\n")
        
        self.test_set = shuffled_df.iloc[: self.test_size]
        self.valid_set = shuffled_df.iloc[self.test_size : self.test_size + self.validation_size]
        self.train_set = shuffled_df.iloc[self.test_size + self.validation_size : ]

        self.train_embeddings = self.train_set[self.features].values
        self.train_labels = self.train_set['track_genre']

    # TC: O(1) - Simply assigns the dataframe.
    # SC: O(n * d) - Space to store the training data embeddings and labels.
    def train(self, df):
        self.train_data = df

    # TC: O(k) - Determines the majority label among the k nearest neighbors.
    # SC: O(k) - Space to store the nearest labels.
    def get_majority(self, nearest_indices):
        nearest_labels = self.train_labels.iloc[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # TC: O(n * d + k) - Calculate distances (O(n * d)) + get majority label (O(k)).
    # SC: O(n) - Space for storing distances.
    def predict_a_sample(self, row):
        start_time = time.time()
        row_embedding = row[self.features].values
        distances = self.calculate_distances(row_embedding)
        nearest_indices = np.argpartition(distances, self.k)[:self.k] # argpartition is more optimized
        prediction = self.get_majority(nearest_indices)
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1
        if self.prediction_count % 10 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(m * (n * d + k)) - Predicting m samples.
    # SC: O(m * n) - Space to store distances for all m test samples.
    def predict(self, X_test):
        self.prediction_count = 0
        self.total_time_taken = 0
        predictions = [self.predict_a_sample(row) for _, row in X_test.iterrows()]
        return predictions
    
"""
    n -> number of training samples
    d -> number of features
    k -> hyperparameter
    m -> number of test samples
    c -> number of metrics
"""
class Best_KNN_model:
    # TC: O(1) => initializes the class variables
    # SC: O(1) => doesn't depend on the size of the input data
    def __init__(self, k: int, distance_metrics: str, features: list):
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_embeddings = None  # Numpy array for training embeddings
        self.train_labels = None  # Numpy array for training labels
        self.prediction_count = 0
        self.total_time_taken = 0

    # TC: O(1) => Simple retrieval of k.
    # SC: O(1) => No additional space is used.
    def get_k(self):
        return self.k
    
    # TC: O(1) => Simple assignment of k.
    # SC: O(1) => No additional space is used.
    def set_k(self, k):
        self.k = k

    # TC: O(1) => Simple retrieval of distance metrics.
    # SC: O(1) => No additional space is used.
    def get_distance_metrics(self):
        return self.distance_metrics
    
    # TC: O(1) => Simple assignment of distance metrics.
    # SC: O(1) => No additional space is used.
    def set_distance_metrics(self, distance_metrics):
        self.distance_metrics = distance_metrics

    # TC: O(nd) => Computes distances between a test sample and all training samples.
    # SC: O(n) => Stores the distances for all training samples.
    def calculate_distances(self, test_embedding):
        if self.distance_metrics == 'manhattan':
            distances = np.sum(np.abs(self.train_embeddings - test_embedding), axis=1)
        elif self.distance_metrics == 'euclidean':
            distances = np.sqrt(np.sum((self.train_embeddings - test_embedding) ** 2, axis=1))
        elif self.distance_metrics == 'cosine':
            dot_product = np.dot(self.train_embeddings, test_embedding)
            norms = np.linalg.norm(self.train_embeddings, axis=1) * np.linalg.norm(test_embedding)
            distances = 1 - (dot_product / norms)
        else:
            raise ValueError("Invalid distance metric")
        return distances
    
    # TC: O(n) => Shuffles and splits the dataset into training, validation, and test sets.
    # SC: O(n) => Stores the embeddings and labels for each split.
    def split_data(self, validation_split=0.1, test_split=0.1):
        self.validation_split = validation_split
        self.test_split = test_split
        total_samples = self.X.shape[0]
        indices = np.random.permutation(total_samples)

        test_size = int(total_samples * test_split)
        valid_size = int(total_samples * validation_split)
        self.test_size = test_size
        self.validation_size = valid_size
        test_indices = indices[:test_size]
        valid_indices = indices[test_size:test_size+valid_size]
        train_indices = indices[test_size+valid_size:]
        
        self.X_test, self.y_test = self.X[test_indices], self.y[test_indices]
        self.X_valid, self.y_valid = self.X[valid_indices], self.y[valid_indices]
        self.X_train, self.y_train = self.X[train_indices], self.y[train_indices]
        
        self.train_embeddings = self.X_train
        self.train_labels = self.y_train

    # TC: O(n) => Converts the input DataFrame into numpy arrays.
    # SC: O(nd) => Stores the features and labels from the DataFrame.
    def train(self, df):
        self.df = df
        self.X = np.array(self.df[self.features].values)
        self.y = np.array(self.df['track_genre'].values)

    # TC: O(k) => Finds the majority label among k nearest neighbors.
    # SC: O(k) => Stores labels of the k nearest neighbors.
    def get_majority(self, nearest_indices):
        nearest_labels = self.train_labels[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # TC: O(nd + k log k) => Computes distances and sorts to find k nearest neighbors.
    # SC: O(n) 
    def predict_a_sample(self, row_embedding):
        start_time = time.time()
        distances = self.calculate_distances(row_embedding)
        nearest_indices = np.argsort(distances)[:self.k] # argpartition is more optimized
        prediction = self.get_majority(nearest_indices)
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1
        if self.prediction_count % 1000 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(m(nd + k log k))
    # SC: O(mn)
    def predict(self, X_test):
        self.prediction_count = 0
        self.total_time_taken = 0
        predictions = [self.predict_a_sample(row) for row in X_test]
        return predictions