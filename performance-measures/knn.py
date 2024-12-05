# Imports necessary for K-NN
import pandas as pd
import numpy as np
from numpy.linalg import norm # Uses: cosine similarity, ...
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from numpy import linalg

class KNN_evaluate:
    def __init__(self, KNN):
        # Initialize the evaluation object with a KNN model
        self.KNN: object = KNN
        self.validation_split = self.KNN.validation_split
        self.test_split = self.KNN.test_split
    
    # calculate the evaluation scores manually using numpy
    def calculate_metrics(self, true_y, pred_y):
        # Get the unique classes in the true labels
        unique_classes = np.unique(true_y) # O(n), Space: O(C) where C is the number of unique classes
        
        # Initialize dictionaries for macro scores
        precision_dict = defaultdict(float) # O(1), Space: O(C)
        recall_dict = defaultdict(float) # O(1), Space: O(C)
        F1_dict = defaultdict(float) # O(1), Space: O(C)
        
        # Initialize variables for micro scores
        tp_micro = 0 # O(1), Space: O(1)
        fp_micro = 0 # O(1), Space: O(1)
        fn_micro = 0 # O(1), Space: O(1)
        
        # Iterate over each unique class to calculate precision, recall, and F1-score
        for cls in unique_classes:
            # Calculate true positives, false positives, and false negatives for each class
            tp = np.sum((true_y == cls) & (pred_y == cls)) # O(n), Space: O(1)
            fp = np.sum((true_y != cls) & (pred_y == cls)) # O(n), Space: O(1)
            fn = np.sum((true_y == cls) & (pred_y != cls)) # O(n), Space: O(1)
            
            # Calculate precision, recall, and F1-score for the current class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # O(1), Space: O(1)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # O(1), Space: O(1)
            F1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # O(1), Space: O(1)   
            
            # Store the calculated metrics in the corresponding dictionaries
            precision_dict[cls] = precision # O(1), Space: O(1)
            recall_dict[cls] = recall # O(1), Space: O(1)
            F1_dict[cls] = F1_score # O(1), Space: O(1)
            
            # for micro scores, accumulate tp, fp, fn
            tp_micro += tp # O(1), Space: O(1)
            fp_micro += fp # O(1), Space: O(1)
            fn_micro += fn # O(1), Space: O(1)
        
        # Calculate macro scores
        macro_precision = np.mean(list(precision_dict.values())) # O(C), Space: O(C)
        macro_recall = np.mean(list(recall_dict.values())) # O(C), Space: O(C)
        macro_F1_score = np.mean(list(F1_dict.values())) # O(C), Space: O(C)
        
        # Calculate micro-average metrics by using accumulated true positives, false positives, and false negatives
        micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        micro_F1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        accuracy = np.mean(true_y == pred_y)
        
        return accuracy, macro_precision, macro_recall, macro_F1_score, micro_precision, micro_recall, micro_F1_score
    
    def evaluate(self, X_test):
        # Extract true labels from the test set
        true_y = X_test['track_genre'].values  # O(f), Space: O(f) where f is the number of samples in X_test
        pred_y = self.KNN.predict(X_test) # O(f * (n * m + n log n + k)), Space: O(f * n)
        
        true_y = np.array(true_y) # O(f), Space: O(f)
        pred_y = np.array(pred_y) # O(f), Space: O(f)

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

    # Print evaluation metrics for a given set (validation or test)
    def print_metrics(self, metrics, set_name):
        print(f"\n{set_name} Set Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


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