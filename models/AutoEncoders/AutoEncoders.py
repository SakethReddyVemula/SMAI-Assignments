import numpy as np
import time
from collections import defaultdict
from models.MLP.MLP import MLP_classifier
import wandb

"""
    n -> number of training samples
    d -> number of features
    k -> hyperparameter
    m -> number of test samples
    c -> number of metrics
"""
class Best_KNN_model:
    # TC: O(1) => [initializes the class variables]
    # SC: O(1) => [doesn't depend on the size of the input data]
    def __init__(self, k: int, distance_metrics: str, features: list):
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_embeddings = None  # Numpy array for training embeddings
        self.train_labels = None  # Numpy array for training labels
        self.prediction_count = 0
        self.total_time_taken = 0

    # TC: O(1)
    # SC: O(1)
    def get_k(self):
        return self.k
    
    # TC: O(1)
    # SC: O(1)
    def set_k(self, k):
        self.k = k

    # TC: O(1)
    # SC: O(1)
    def get_distance_metrics(self):
        return self.distance_metrics
    
    # TC: O(1)
    # SC: O(1)
    def set_distance_metrics(self, distance_metrics):
        self.distance_metrics = distance_metrics

    # TC: O(nd)
    # SC: O(n)
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
    
    # TC: O(n)
    # SC: O(n)
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

    # TC: O(n)
    # SC: O(nd)
    def train(self, df):
        self.df = df
        self.X = np.array(self.df[self.features].values)
        self.y = np.array(self.df['track_genre'].values)

    # TC: O(k)
    # SC: O(k)
    def get_majority(self, nearest_indices):
        nearest_labels = self.train_labels[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # TC: O(nd + k log k)
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


class AutoEncoder:
    def __init__(self, input_dim: int=10, latent_dim: int=3, hidden_layers=1, hidden_dim=32, lr: float=1e-3, act_type="sigmoid", optimizer_type="mini", batch_size=32, max_iter=5000):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.l_rate = lr
        self.activation_type = act_type
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.max_iterations = max_iter
        
        # Encoder (10 -> 3)
        self.encoder = MLP_classifier(
            input_dim=input_dim,
            output_dim=latent_dim,
            n_hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            lr=lr,
            act_type=act_type,
            optimizer_type=optimizer_type,
            batch_size=batch_size,
            max_iter=max_iter,
            objective="regression"
        )
        
        # Decoder (3 -> 10)
        self.decoder = MLP_classifier(
            input_dim=latent_dim,
            output_dim=input_dim,
            n_hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            lr=lr,
            act_type=act_type,
            optimizer_type=optimizer_type,
            batch_size=batch_size,
            max_iter=max_iter,
            objective="regression"
        )

    def forward(self, X: np.ndarray):
        latent = self.encoder.forward(X)
        return self.decoder.forward(latent)
    
    def compute_loss(self, X: np.ndarray):
        reconstructed = self.forward(X)
        return np.mean((X - reconstructed) ** 2)

    def compute_gradients(self, X):
        # [16, 10] -> [16, 15] -> [16, 3] -> [16, 15] -> [16, 10]
        # Forward pass
        latent = self.encoder.forward(X)
        # print(f"latent shape: {latent.shape}")
        reconstructed = self.decoder.forward(latent)
        # print(f"reconstructed shape: {reconstructed.shape}")
        
        # Compute reconstruction error
        error = (reconstructed - X)
        # print(f"error shape: {error.shape}")
        
        # Backpropagation through decoder
        decoder_delta = error
        decoder_gradients = []
        for layer in reversed(range(len(self.decoder.weights))):
            # print(f"Decoder layer {layer}:")
            # print(f"  activation shape: {self.decoder.activations[layer].shape}")
            # print(f"  weights shape: {self.decoder.weights[layer].shape}")
            # print(f"  delta shape: {decoder_delta.shape}")
            
            d_weights = np.dot(self.decoder.activations[layer].T, decoder_delta)
            decoder_gradients.insert(0, d_weights)
            
            # if layer > 0: ( not required since atleast once we need to backpropagate through the boundary)
            decoder_delta = np.dot(decoder_delta, self.decoder.weights[layer].T) * self.decoder.activation_function_prime(self.decoder.activations[layer])
            # print(f"decoder_delta_shape: {decoder_delta.shape}")
        # The gradient at the input of the decoder is the gradient w.r.t. the latent representation
        d_latent = decoder_delta
        
        # Backpropagation through encoder
        encoder_delta = d_latent
        # print(f"encoder_delta shape: {encoder_delta.shape}")

        encoder_gradients = []
        for layer in reversed(range(len(self.encoder.weights))):
            # print(f"Encoder layer {layer}:")
            # print(f"  activation shape: {self.encoder.activations[layer].shape}")
            # print(f"  weights shape: {self.encoder.weights[layer].shape}")
            # print(f"  delta shape: {encoder_delta.shape}")
            
            d_weights = np.dot(self.encoder.activations[layer].T, encoder_delta)
            encoder_gradients.insert(0, d_weights)
            
            if layer > 0:
                encoder_delta = np.dot(encoder_delta, self.encoder.weights[layer].T) * self.encoder.activation_function_prime(self.encoder.activations[layer])
        # print(f"encoder_gradients_shape: {encoder_gradients[0].shape}")
        # print(f"decoder_gradients_shape: {decoder_gradients[0].shape}")

        formatted_encoder_gradients = np.concatenate([grad.ravel() for grad in encoder_gradients])
        formatted_decoder_gradients = np.concatenate([grad.ravel() for grad in decoder_gradients])

        return formatted_encoder_gradients, formatted_decoder_gradients
    
    def update(self, encoder_gradients, decoder_gradients):
        self.encoder.update(encoder_gradients)
        self.decoder.update(decoder_gradients)

    def fit(self, X: np.ndarray, X_val: np.ndarray=None, wandb_log: bool=False, max_iterations: int=None):
        np.random.shuffle(X)

        if X_val is None:
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
        else:
            X_train = X

        # print(f"X_train_shape: {X_train.shape}")
        # print(f"X_val_shape: {X_val.shape}")
        batch_size = self.encoder.batch_size
        max_iter = max_iterations if max_iterations is not None else self.encoder.max_iter

        for epoch in range(max_iter):
            # shuffle the data
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i: i + batch_size]
                X_batch = X_train[batch_indices]

                # compute gradients
                encoder_gradients, decoder_gradients = self.compute_gradients(X_batch)

                # update weights
                self.update(encoder_gradients, decoder_gradients)

            # compute and log losses
            train_loss = self.compute_loss(X_train)
            val_loss = self.compute_loss(X_val)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}\tTrain Loss: {train_loss:.4f}\tValidation Loss: {val_loss:4f}")

            if wandb_log == True:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "validation/loss": val_loss
                })

        print(f"AutoEncoder training complete\n")
    
    def get_latent(self, X: np.ndarray):
        return self.encoder.forward(X)
    
    def reconstruct(self, X: np.ndarray):
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray):
        mse = self.compute_loss(X)
        print(f"Mean Squared Error: {mse}")
        return mse