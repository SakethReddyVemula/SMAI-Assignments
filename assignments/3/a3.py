# Import Libraries
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import pickle
import os
from datetime import datetime

class MLP_classifier:
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            n_hidden_layers: int = 3, 
            hidden_dim: int = 32, 
            lr: float = 1e-4, 
            act_type: str="sigmoid", 
            optimizer_type: str="mini", 
            batch_size: int=8, 
            max_iter: int=10000, 
            do_gradient_check:bool = False,
            do_multi_label: bool = False,
            objective: str = "classification"
        ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.act_type = act_type
        self.optimizer_type = optimizer_type
        self.n_hidden_layers = n_hidden_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.set_activation_function() # Initialize according to provided argument
        self.set_optimizer(optimizer_type)
        self.weights = self.initialize_weights() # Initialize the weights upon class initialization itself
        self.do_gradient_check = do_gradient_check
        self.do_multi_label = do_multi_label
        self.objective = objective
        
    def get_lr(self) -> float:
        return self.lr

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def get_max_iter(self) -> int:
        return self.max_iter
    
    def set_max_iter(self, max_iter: int) -> None:
        self.max_iter = max_iter
    
    def get_act_type(self) -> str:
        return self.act_type
    
    def set_act_type(self, act_type: str) -> None:
        self.act_type = act_type
        self.activation_function()

    def get_optimizer(self) -> str:
        return self.optimizer_type
    
    def set_optimizer(self, optimizer_type: str) -> None:
        if optimizer_type == "batch":
            self.train_function = self.train_batch
        elif optimizer_type == "mini":
            self.train_function = self.train_mini_batch
        elif optimizer_type == "sgd":
            self.train_function = self.train_sgd
        else:
            raise(ValueError("Invalid optimizer type"))
        

    def get_n_hidden_layers(self) -> int:
        return self.n_hidden_layers
    
    def set_n_hidden_layers(self, n_hidden_layers: int) -> None:
        self.n_hidden_layers = n_hidden_layers

    def get_hidden_dim(self) -> int:
        return self.hidden_dim 
    
    def set_hidden_dim(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim

    def linear(self, z):
        return z
    
    def linear_prime(self, z):
        return np.ones_like(z.shape[1])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return (self.sigmoid(z)) * (1 - (self.sigmoid(z)))
    
    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def tanh_prime(self, z):
        return (1 - np.square(self.tanh(z)))

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_prime(self, z) -> float:
        return np.where(z > 0, 1.0, 0.0)
    
    def leakyrelu(self, z, neg_slope: float=0.01):
        return np.maximum(neg_slope * z, z)
    
    def leakyrelu_prime(self, z, neg_slope: float=0.01) -> float:
        return np.where(z > 0, 1.0, neg_slope)
    
    def set_activation_function(self):
        if self.act_type == "sigmoid":
            self.activation_function = self.sigmoid
            self.activation_function_prime = self.sigmoid_prime
        elif self.act_type == "tanh":
            self.activation_function = self.tanh
            self.activation_function_prime = self.tanh_prime
        elif self.act_type == "relu":
            self.activation_function = self.relu
            self.activation_function_prime = self.relu_prime
        elif self.act_type == "leakyrelu":
            self.activation_function = self.leakyrelu
            self.activation_function_prime = self.leakyrelu_prime
        elif self.act_type == "linear":
            self.activation_function = self.linear
            self.activation_function_prime = self.linear_prime
        else:
            raise(ValueError("Invalid activation function"))
        
    def initialize_weights(self):
        weights = []
        layer_sizes = [self.input_dim] + [self.hidden_dim] * self.n_hidden_layers + [self.output_dim]
        for i in range(1, len(layer_sizes)):
            weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]))
        return weights
        
    def softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis = 1, keepdims=True)
        
    def forward(self, X: np.ndarray):
        # print(f"Forward input shape: {X.shape}")
        activated = [X]
        pre_activated = []
        for layer in range(0, self.n_hidden_layers):
            # print(f"acivated[layer]_shape: {activated[layer].shape}")
            # print(f"self.weights[layer]_shape: {self.weights[layer].shape}")
            z = np.dot(activated[layer], self.weights[layer]) # Z' = A * W
            a = self.activation_function(z) # A' = activation_function(Z')
            pre_activated.append(z)
            activated.append(a)

        z = np.dot(activated[self.n_hidden_layers], self.weights[self.n_hidden_layers])
        pre_activated.append(z)
        if self.objective == "classification":
            self.y_ht = self.softmax(pre_activated[-1]) # Classification use the softmaxed output (since we deal with probabilities)
        else:
            self.y_ht = pre_activated[-1] # Regression: directly use raw output
        activated.append(self.y_ht)
        self.activations = activated # A's
        self.weighted_sums = pre_activated # Z's
        # print(f"Forward output shape: {self.y_ht.shape}")
        return self.y_ht
        
    # Classification optimizes cross entropy loss
    # def cost_function_classification(self, X: np.ndarray, y: np.ndarray):
    #     self.y_ht = self.forward(X)
    #     loss = (-1) * np.sum((y * np.log(self.y_ht)) / len(X)) # Usign log for stability (avoids underflow)
    #     return loss
    
    # # Cross Entropy
    # def cost_function_classification(self, X: np.ndarray, y: np.ndarray):
    #     epsilon = 1e-15  # To avoid log(0)
    #     self.y_ht = self.forward(X)  # self.y_ht should be the predicted probabilities
    #     # Compute cross-entropy loss
    #     loss = - np.mean(y * np.log(self.y_ht + epsilon) + (1 - y) * np.log(1 - self.y_ht + epsilon))
    #     return loss
    
    # KL-Divergence Loss
    def cost_function_classification(self, X: np.ndarray, y: np.ndarray):
        epsilon = 1e-15  # To avoid log(0)
        self.y_ht = self.forward(X)  # self.y_ht should be the predicted probabilities
        # Compute KL-Divergence loss
        loss = np.sum(y * np.log((y + epsilon) / (self.y_ht + epsilon))) / len(X)
        return loss

    # Regression optimizes Mean Square Error
    def cost_function_regression(self, X: np.ndarray, y: np.ndarray):
        self.y_ht = self.forward(X)
        loss = np.square(y - self.y_ht).mean() # better numerical stability
        return loss
    
    def backward(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]
        # print(f"y_shape: {y.shape}")
        # print(f"self.activations[-1]_shape: {self.activations[-1].shape}")
        
        # Initialize gradients list
        gradients = [np.zeros_like(w) for w in self.weights]
        
        # Output layer error
        if self.objective == "classification":
            delta = self.activations[-1] - y  # For softmax with cross-entropy, this is the gradient
        elif self.objective == "regression":
            delta = 2 * (self.activations[-1] - y) / m
        
        # Calculate gradient for the last layer
        gradients[-1] = np.dot(self.activations[-2].T, delta) / m
        
        # Backpropagate through hidden layers
        for layer in reversed(range(self.n_hidden_layers)):
            # Propagate error
            # print(f"delta_shape: {delta.shape}")
            # print(f"self.weights[layer+1].T: {self.weights[layer+1].T.shape}")
            # print(f"self.weighted_sums[layer]: {self.weighted_sums[layer].shape}")
            delta = np.dot(delta, self.weights[layer+1].T) * self.activation_function_prime(self.weighted_sums[layer])
            
            # Calculate gradient
            gradients[layer] = np.dot(self.activations[layer].T, delta) / m
        
        return gradients
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray):
        # Forward Propagation
        self.forward(X) # (n_samples, output_dim)
        # Backward Propagation
        gradients = self.backward(X, y)
        return np.concatenate([grad.ravel() for grad in gradients])

    # Update Steps
    def get_params(self):
        return np.concatenate([w.ravel() for w in self.weights])
    
    def set_params(self, params):
        start = 0
        for i in range(len(self.weights)):
            end = start + self.weights[i].size
            self.weights[i] = params[start:end].reshape(self.weights[i].shape)
            start = end
    
    def update(self, gradients):
        # print(f"Shape of gradients (input to Update): {gradients.shape}")
        params = self.get_params()
        updated_params = params - self.lr * gradients
        self.set_params(updated_params)

    # Note: all changes needed to hyperparameters must be done manually before calling train_batch
    def train_batch(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, wandb_log = False, max_iterations=None, batch_size:int = None):
        if self.do_gradient_check:
            self.run_gradient_check(X, y)
        
        self.losses = []
        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
        print(f"Initial Loss:\t{loss}")
        if max_iterations is not None:
            self.set_max_iter(max_iterations) # Set max_iter to max_iterations

        epochs_trained = 0
        for epoch in range(self.max_iter):
            if ((epoch >= 3) and (self.losses[-2] - self.losses[-1]) <= 1e-5):
                print(f"Early Stopping at iteration: {epoch}")
                break
            
            gradients = self.compute_gradients(X, y)
            # print(f"gradients shape: {gradients.shape}")
            if self.objective == "classification":
                loss = self.cost_function_classification(X, y)
            elif self.objective == "regression":
                loss = self.cost_function_regression(X, y)
            self.losses.append(loss)
            self.update(gradients)

            # test loss
            if self.objective == "classification":
                loss_val = self.cost_function_classification(X_val, y_val)
            elif self.objective == "regression":
                loss_val = self.cost_function_regression(X_val, y_val)
            if epoch % 100 == 0:
                print(f"Epoch:\t{epoch}\ttrain/loss:\t{loss}\tvalidation/loss:\t{loss_val}")
            if self.objective == "classification":
                a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch, 
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/accuracy": a_s,
                        "validation/precision": p_s,
                        "validation/recall": r_s,
                        "validation/f1_score": f_s
                    })
                epochs_trained += 1
            elif self.objective == "regression":
                mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/mse": mse,
                        "validation/rmse": rmse,
                        "validation/r_squared": r_squared
                    })
                epochs_trained += 1

        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
            loss_val = self.cost_function_classification(X_val, y_val)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
            loss_val = self.cost_function_regression(X_val, y_val)
        print(f"Batch training successful")
        print(f"Total Epochs: {epochs_trained}")
        print(f"Final Train Loss: {loss}")
        print(f"Final Validation Loss: {loss_val}")
        if self.objective == "classification":
            a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch, 
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/accuracy": a_s,
                    "validation/precision": p_s,
                    "validation/recall": r_s,
                    "validation/f1_score": f_s
                })
        elif self.objective == "regression":
            mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/mse": mse,
                    "validation/rmse": rmse,
                    "validation/r_squared": r_squared
                })

        return loss
    
    def train_mini_batch(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, wandb_log=False, max_iterations=None, batch_size: int=32):
        if self.do_gradient_check:
            self.run_gradient_check(X, y)
        self.losses = []
        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
        print(f"Initial Loss:\t{loss}")
        if max_iterations is not None:
            self.set_max_iter(max_iterations)

        epochs_trained = 0
        for epoch in range(self.max_iter):
            # if ((epoch >= 3) and (self.losses[-3] - self.losses[-1]) <= 1e-5):
            #     print(f"Early Stopping at iteration: {epoch}")
            #     break
            
            # make batches from 1 sized batches
            indices = np.arange(len(X))
            # print(f"indices: {indices}")
            np.random.shuffle(indices)
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i: i + batch_size]
                gradients = self.compute_gradients(X[batch_indices], y[batch_indices])
                self.update(gradients)
            if self.objective == "classification":
                loss = self.cost_function_classification(X, y)
            elif self.objective == "regression":
                loss = self.cost_function_regression(X, y)
            self.losses.append(loss)

            # Validation loss
            if self.objective == "classification":
                loss_val = self.cost_function_classification(X_val, y_val)
            elif self.objective == "regression":
                loss_val = self.cost_function_regression(X_val, y_val)
            if epoch % 1 == 0:
                print(f"Epoch:\t{epoch}\ttrain/loss:\t{loss}\tvalidation/loss:\t{loss_val}")
            if self.objective == "classification":
                a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch, 
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/accuracy": a_s,
                        "validation/precision": p_s,
                        "validation/recall": r_s,
                        "validation/f1_score": f_s
                    })
                epochs_trained += 1
            elif self.objective == "regression":
                mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/mse": mse,
                        "validation/rmse": rmse,
                        "validation/r_squared": r_squared
                    })
                epochs_trained += 1

        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
            loss_val = self.cost_function_classification(X_val, y_val)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
            loss_val = self.cost_function_regression(X_val, y_val)
        print(f"Mini-Batch training successful")
        print(f"Total Epochs: {epochs_trained}")
        print(f"Final Train Loss: {loss}")
        print(f"Final Validation Loss: {loss_val}")
        if self.objective == "classification":
            a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch, 
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/accuracy": a_s,
                    "validation/precision": p_s,
                    "validation/recall": r_s,
                    "validation/f1_score": f_s
                })
        elif self.objective == "regression":
            mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/mse": mse,
                    "validation/rmse": rmse,
                    "validation/r_squared": r_squared
                })

        return loss
    
    def train_sgd(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, wandb_log=False, max_iterations=None, batch_size: int=None):
        batch_size = 1
        EARLY_THRESHOLD = 1e-5

        if self.do_gradient_check:
            self.run_gradient_check(X, y)
        self.losses = []
        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
        print(f"Initial Loss:\t{loss}")
        if max_iterations is not None:
            self.set_max_iter(max_iterations)

        epochs_trained = 0
        val_loss_list = []

        for epoch in range(self.max_iter):
            # make batches from 1 sized batches
            indices = np.arange(len(X))
            # print(f"indices: {indices}")
            np.random.shuffle(indices)
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i: i + batch_size]
                gradients = self.compute_gradients(X[batch_indices], y[batch_indices])
                self.update(gradients)

            if self.objective == "classification":
                loss = self.cost_function_classification(X, y)
            elif self.objective == "regression":
                loss = self.cost_function_regression(X, y)
            self.losses.append(loss)

            # Validation loss
            if self.objective == "classification":
                loss_val = self.cost_function_classification(X_val, y_val)
            elif self.objective == "regression":
                loss_val = self.cost_function_regression(X_val, y_val)
            val_loss_list.append(loss_val)

            if epoch % 100 == 0:
                print(f"Epoch:\t{epoch}\ttrain/loss:\t{loss}\tvalidation/loss:\t{loss_val}")
            if self.objective == "classification":
                a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch, 
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/accuracy": a_s,
                        "validation/precision": p_s,
                        "validation/recall": r_s,
                        "validation/f1_score": f_s
                    })
                epochs_trained += 1
            elif self.objective == "regression":
                mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
                if wandb_log == True:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": loss,
                        "validation/loss": loss_val,
                        "validation/mse": mse,
                        "validation/rmse": rmse,
                        "validation/r_squared": r_squared
                    })
                epochs_trained += 1
            # if epoch >= 2:
            #     if (self.losses[-2] - self.losses[-1]) <= EARLY_THRESHOLD:
            #         print(f"Early Stopping at epoch: {epoch}")
            #         break

        if self.objective == "classification":
            loss = self.cost_function_classification(X, y)
            loss_val = self.cost_function_classification(X_val, y_val)
        elif self.objective == "regression":
            loss = self.cost_function_regression(X, y)
            loss_val = self.cost_function_regression(X_val, y_val)
        print(f"SGD training successful")
        print(f"Total Epochs: {epochs_trained}")
        print(f"Final Train Loss: {loss}")
        print(f"Final Validation Loss: {loss_val}")
        if self.objective == "classification":
            a_s, p_s, r_s, f_s = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch, 
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/accuracy": a_s,
                    "validation/precision": p_s,
                    "validation/recall": r_s,
                    "validation/f1_score": f_s
                })
        elif self.objective == "regression":
            mse, rmse, r_squared = self.evaluate_validation(X_val, y_val)
            if wandb_log == True:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": loss,
                    "validation/loss": loss_val,
                    "validation/mse": mse,
                    "validation/rmse": rmse,
                    "validation/r_squared": r_squared
                })

        return loss

    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, wandb_log: bool=False, max_iterations: int=None, batch_size: int=None):
        if self.optimizer_type != "mini":
            self.train_function(X, y, X_val, y_val, wandb_log=wandb_log, max_iterations=max_iterations, batch_size=batch_size)
        else:
            self.train_function(X, y, X_val, y_val, wandb_log=wandb_log, max_iterations=max_iterations)
        
    def predict(self, X: np.ndarray):
        if self.do_multi_label == False:
            y_ht = self.forward(X)
            predictions = np.zeros_like(y_ht)
            predictions[np.arange(len(y_ht)), y_ht.argmax(axis=1)] = 1
            return predictions
        else:
            y_ht = self.forward(X)
            binary_predictions = (y_ht > 0.125).astype(int)
            return binary_predictions


    def evaluate_validation(self, X_val_proc: np.ndarray, y_val: np.ndarray):
        if self.objective == "classification":
            y_pred = self.predict(X_val_proc)
            acc_score = accuracy_score(y_val, y_pred)
            prec_score = precision_score(y_val, y_pred, average='macro', zero_division=0)
            rec_score = recall_score(y_val, y_pred, average='macro', zero_division=0)
            f1_s = f1_score(y_val, y_pred, average='micro', zero_division=0)
            return acc_score, prec_score, rec_score, f1_s
        elif self.objective == "regression":
            y_pred = self.forward(X_val_proc)
            mse = np.mean((y_val - y_pred) ** 2)
            rmse = np.sqrt(mse)
            y_mean = np.mean(y_val)
            total_sum_squares = np.sum((y_val - y_mean) ** 2)
            residual_sum_squares = np.sum((y_val - y_pred) ** 2)
            r_squared = 1 - (residual_sum_squares / total_sum_squares)
            return mse, rmse, r_squared

    def evaluate(self, X_val_proc: np.ndarray, X_test_proc: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        if self.objective == "classification":
            if self.do_multi_label == False:
                print(f"Performance measure on Single label classification")
                y_pred = self.predict(X_val_proc)
                print(f"-"*90)
                print(f"Evaluation on Validation set")
                print(f"Accuracy:\t{accuracy_score(y_val, y_pred)}")
                print(f"Precision:\t{precision_score(y_val, y_pred, average='macro', zero_division=0)}")
                print(f"Recall:\t{recall_score(y_val, y_pred, average='macro', zero_division=0)}")
                print(f"F1-Score:\t{f1_score(y_val, y_pred, average='micro', zero_division=0)}")
                print(f"-"*90)
                y_pred = self.predict(X_test_proc)
                acc_score = accuracy_score(y_test, y_pred)
                prec_score = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec_score = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1_s = f1_score(y_test, y_pred, average='micro', zero_division=0)
                print(f"Evaluation on Test set")
                print(f"Accuracy:\t{acc_score}")
                print(f"Precision:\t{prec_score}")
                print(f"Recall:\t{rec_score}")
                print(f"F1-Score:\t{f1_s}")
                print(f"-"*90)
                return acc_score, prec_score, rec_score, f1_s
            else:
                print(f"Performance measure on Multi label classification")
                y_pred = self.predict(X_val_proc)
                print("True validation:")
                print(y_val)
                print("Binary True Validation")
                binary_y_val = (y_val > 0.125).astype(int)
                print(binary_y_val)
                print("Predicted Validation")
                print(y_pred)
                print(f"-"*90)
                print(f"Evaluation on Validation set")
                print(f"Hamming Loss:\t{hamming_loss(binary_y_val, y_pred)}")
                print(f"Accuracy:\t{accuracy_score(binary_y_val, y_pred)}")
                print(f"Precision:\t{precision_score(binary_y_val, y_pred, average='samples', zero_division=0)}")
                print(f"Recall:\t{recall_score(binary_y_val, y_pred, average='samples', zero_division=0)}")
                print(f"F1-Score:\t{f1_score(binary_y_val, y_pred, average='samples', zero_division=0)}")
                print(f"-"*90)
                y_pred = self.predict(X_test_proc)
                print(f"Evaluation on Test set")
                print(f"Hamming Loss:\t{hamming_loss(y_test, y_pred)}")
                acc_score = accuracy_score(y_test, y_pred)
                print(f"Accuracy:\t{acc_score}")
                prec_score = precision_score(y_test, y_pred, average='samples', zero_division=0)
                print(f"Precision:\t{prec_score}")
                rec_score = recall_score(y_test, y_pred, average='samples', zero_division=0)
                print(f"Recall:\t{rec_score}")
                f1_s = f1_score(y_test, y_pred, average='samples', zero_division=0)
                print(f"F1-Score:\t{f1_s}")
                print(f"-"*90)
                return acc_score, prec_score, rec_score, f1_s
        elif self.objective == "regression":
            print(f"Performance measure on Regression")
            
            # Validation set evaluation
            y_pred_val = self.forward(X_val_proc)
            mse_val = np.mean((y_val - y_pred_val) ** 2)
            rmse_val = np.sqrt(mse_val)
            y_val_mean = np.mean(y_val)
            val_total_sum_squares = np.sum((y_val - y_val_mean) ** 2)
            val_residual_sum_squares = np.sum((y_val - y_pred_val) ** 2)
            r_squared_val = 1 - (val_residual_sum_squares / val_total_sum_squares)
            
            print(f"-"*90)
            print(f"Evaluation on Validation set")
            print(f"Mean Squared Error (MSE):\t{mse_val:.6f}")
            print(f"Root Mean Squared Error (RMSE):\t{rmse_val:.6f}")
            print(f"R-squared (R²):\t{r_squared_val:.6f}")
            print(f"-"*90)
            
            # Test set evaluation
            y_pred_test = self.forward(X_test_proc)
            mse_test = np.mean((y_test - y_pred_test) ** 2)
            rmse_test = np.sqrt(mse_test)
            y_test_mean = np.mean(y_test)
            test_total_sum_squares = np.sum((y_test - y_test_mean) ** 2)
            test_residual_sum_squares = np.sum((y_test - y_pred_test) ** 2)
            r_squared_test = 1 - (test_residual_sum_squares / test_total_sum_squares)
            
            print(f"Evaluation on Test set")
            print(f"Mean Squared Error (MSE):\t{mse_test:.6f}")
            print(f"Root Mean Squared Error (RMSE):\t{rmse_test:.6f}")
            print(f"R-squared (R²):\t{r_squared_test:.6f}")
            print(f"-"*90)
            
            return mse_test, rmse_test, r_squared_test
                

    """
        Gradients checking: https://medium.com/farmart-blog/understanding-backpropagation-and-gradient-checking-6a5c0ba73a68
        https://cs231n.github.io/neural-networks-3/
    """

    def compute_numerical_gradients(self, X: np.ndarray, y: np.ndarray, epsilon: float=1e-7):
        params = self.get_params()
        numerical_gradients = np.zeros_like(params)

        for idx in range(len(params)):
            # create small perturbations
            params_plus = params.copy()
            params_plus[idx] += epsilon
            params_minus = params.copy()
            params_minus[idx] -= epsilon

            # compute cost for both perturbations
            self.set_params(params_plus)
            if self.objective == "classification":
                cost_plus = self.cost_function_classification(X, y)
            elif self.objective == "regression":
                cost_plus = self.cost_function_regression(X, y)

            self.set_params(params_minus)
            if self.objective == "classification":
                cost_minus = self.cost_function_classification(X, y)
            elif self.objective == "regression":
                cost_minus = self.cost_function_regression(X, y)

            # compute numerical gradients
            numerical_gradients[idx] = (cost_plus - cost_minus) / (2 * epsilon)

        # Restore original params
        self.set_params(params)
        return numerical_gradients
    
    def check_gradients(self, X: np.ndarray, y: np.ndarray, epsilon: float=1e-5, tolerance: float=1e-6):
        """
            relative error > 1e-2 usually means the gradient is probably wrong
            1e-2 > relative error > 1e-4 should make you feel uncomfortable
            1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.
            1e-7 and less you should be happy.
        """
        SMALL_EPSILON = 1e-15
        analytical_gradients = self.compute_gradients(X, y)
        # print(f"analytical gradients: {analytical_gradients}")
        numerical_gradients = self.compute_numerical_gradients(X, y, epsilon)
        # print(f"numerical gradients: {numerical_gradients}")

        absolute_diff = np.abs(analytical_gradients - numerical_gradients)
        denominator = np.abs(analytical_gradients) + np.abs(numerical_gradients)
        relative_error = np.where(denominator > 0, (absolute_diff + SMALL_EPSILON) / (denominator + SMALL_EPSILON), 0)
        # relative_error = (absolute_diff + SMALL_EPSILON) / (denominator + SMALL_EPSILON)
        # print(f"relative error: {relative_error}")
        
        max_relative_error = np.max(relative_error)

        is_correct = max_relative_error < tolerance

        return is_correct, max_relative_error
    
    def run_gradient_check(self, X: np.ndarray, y: np.ndarray, n_samples: int = 10):
        print("Running gradient check...")
        
        # Use a small subset of data for gradient checking
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        is_correct, max_error = self.check_gradients(X_subset, y_subset)
        
        print(f"Gradient check {'passed' if is_correct else 'failed'}!")
        print(f"Maximum relative error: {max_error:.2e}")
        
        if not is_correct:
            print("Warning: Gradient check failed! Implementation might be incorrect.")
        else:
            print("Gradient implementation looks correct!")

    def save_model(self, filepath: str = None) -> str:
        if filepath is None:
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"mlp_model_{timestamp}.pkl"
        
        # Create a dictionary of model parameters
        model_state = {
            'weights': self.weights,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_hidden_layers': self.n_hidden_layers,
            'hidden_dim': self.hidden_dim,
            'lr': self.lr,
            'act_type': self.act_type,
            'optimizer_type': self.optimizer_type,
            'batch_size': self.batch_size,
            'max_iter': self.max_iter
        }
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        print(f"Model saved successfully to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLP_classifier':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")
        
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create a new instance with loaded parameters
        model = cls(
            input_dim=model_state['input_dim'],
            output_dim=model_state['output_dim'],
            n_hidden_layers=model_state['n_hidden_layers'],
            hidden_dim=model_state['hidden_dim'],
            lr=model_state['lr'],
            act_type=model_state['act_type'],
            optimizer_type=model_state['optimizer_type'],
            batch_size=model_state['batch_size'],
            max_iter=model_state['max_iter']
        )
        
        # Load the weights
        model.weights = model_state['weights']
        
        print(f"Model loaded successfully from {filepath}")
        return model
    
def train_and_save_best_model(X_train, y_train, X_val, y_val, best_params):
    # Create model with best parameters
    best_model = MLP_classifier(
        input_dim=X_train.shape[1],
        output_dim=len(np.unique(y_train)),
        n_hidden_layers=best_params['hidden_layers'],
        hidden_dim=best_params['hidden_dim'],
        lr=best_params['lr'],
        act_type=best_params['activation_type'],
        optimizer_type=best_params['optimizer_type']
    )
    
    # Train the model
    best_model.fit(X_train, y_train, X_val, y_val)
    
    # Save the model
    saved_model_path = best_model.save_model()
    return saved_model_path

def load_and_evaluate_model(model_path, X_val, y_val, X_test, y_test):
    # Load the model
    loaded_model = MLP_classifier.load_model(model_path)
    
    # Evaluate the model
    loaded_model.evaluate(X_val, X_test, y_val, y_test)

#######################################################################################################
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



#######################################################################################################

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import wandb
# from mlp_classifier import MLP_classifier, train_and_save_best_model, load_and_evaluate_model
# from AutoEncoder import AutoEncoder, Best_KNN_evaluate, Best_KNN_model
# from mlp_classifier import ml_mlp_classifier
import sys

def task_2_1():
    df = pd.read_csv("WineQT.csv")
    df.hist(bins = 20, figsize=(15, 10))
    plt.tight_layout()
    plt.savefig("figures/2_1_hist.png")

    tuples = []
    for column in df.columns:
        column_data = df[column]
        mean = np.mean(column_data)
        min = np.min(column_data)
        max = np.max(column_data)
        standard_deviation = np.std(column_data)
        tuples.append((mean,standard_deviation,max,min))

    df_metrics = pd.DataFrame(tuples, columns=['mean', 'standard_deviation', 'max', 'min'], index=df.columns)

    quality_counts = df['quality'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Quality')
    plt.savefig("figures/2_1_pie_chart.png")

    # read the data
    df = pd.read_csv("WineQT.csv")
    df.dropna(inplace=True)
    df = df.drop(['Id'], axis=1)
    df_copy = df


    X = df_copy.drop('quality', axis=1)
    y = df_copy['quality']
    unique_output_vals = len(np.unique(y))
    print(y)
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # If you have only 2 classes, LabelBinarizer will return a 1D array
    # To make it consistent with multi-class case, reshape it:
    if len(lb.classes_) == 2:
        y = np.hstack((1 - y, y))
    
    print(y)

    # Split
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.33, random_state=42, shuffle=True)

    # Print n_samples in each
    print(f"num_samples in X_train: {len(X_train)}")
    print(f"num_samples in X_val: {len(X_val)}")
    print(f"num_samples in X_test: {len(X_test)}")

    # Intialize the scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # # Standardize the normalized data
    X_train_standardized = standard_scaler.fit_transform(X_train)
    X_test_standardized = standard_scaler.fit_transform(X_test)
    X_val_standardized = standard_scaler.fit_transform(X_val)

    # Normalize the data using Min-Max scaling

    # On normal data
    X_train_normalized = min_max_scaler.fit_transform(X_train)
    X_val_normalized = min_max_scaler.fit_transform(X_val)
    X_test_normalized = min_max_scaler.fit_transform(X_test)

    # Normalize standardized data
    # X_train_normalized = min_max_scaler.fit_transform(X_train_standardized)
    # X_val_normalized = min_max_scaler.fit_transform(X_val_standardized)
    # X_test_normalized = min_max_scaler.fit_transform(X_test_standardized)

    # Convert the scaled arrays back to DataFrames
    X_train_processed = pd.DataFrame(X_train_normalized, columns=X.columns)
    X_test_processed = pd.DataFrame(X_test_normalized, columns=X.columns)
    X_val_processed = pd.DataFrame(X_val_normalized, columns=X.columns)

    # X_train_processed = pd.DataFrame(X_train_standardized, columns=X.columns)
    # X_test_processed = pd.DataFrame(X_test_standardized, columns=X.columns)
    # X_val_processed = pd.DataFrame(X_val_standardized, columns=X.columns)

    # Function to calculate and print statistics
    def print_stats(data, name):
        # print(f"\n{name} (first 5 rows):")
        # print(data.head())
        print(f"\n{name} statistics:")
        print(data.describe().loc[['mean', 'std', 'min', 'max']].T)

    # Print statistics for original and processed data
    print_stats(X_train, "Original data")

    # Verify the range of original data
    print("\nRange of original data:")
    print(f"Min: {X_train.min().min()}")
    print(f"Max: {X_train.max().max()}")

    print_stats(X_train_processed, "Normalized data")

    # Verify the range of processed data
    print("\nRange of processed data:")
    print(f"Min: {X_train_processed.min().min()}")
    print(f"Max: {X_train_processed.max().max()}")

def task_2_2(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim:int, wandb_log=False):
    wandb_log = wandb_log
    L_RATE = 1e-3
    MAX_ITERATIONS = 5000
    OPTIMIZER_TYPE = "mini"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=1


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS, 
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True
    )



    # Print the shapes of each weights (to debug)
    for i in range(NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)

def task_2_3(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False):
    l_rate_list = [1e-3, 5e-4, 1e-4]
    # optimizer_type_list = ["batch", "mini", "sgd"]
    optimizer_type_list = ["sgd"]
    activation_type_list = ["linear", "sigmoid", "tanh", "relu"]
    hidden_dim_list = [11, 15, 20]
    hidden_layers_list = [1, 2]

    results = []

    sweep_config = {
        "method": "grid",
        "name": "Hyperparameter_Tuning",
        "metric": {
            "goal": "maximize",
            "name": "validation/accuracy"
        },
        "parameters": {
            "l_rate": {
                "values": l_rate_list
            },
            "optimizer_type": {
                "values": optimizer_type_list
            },
            "activation_type": {
                "values": activation_type_list
            },
            "hidden_dim": {
                "values": hidden_dim_list
            },
            "hidden_layers": {
                "values": hidden_layers_list
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project = "SMAI-A3-MLP"
    )

    def tune_model():
        wandb.init(
            project = "SMAI-A3-MLP"
        )
        config = wandb.config
        l_rate = config.l_rate
        optimizer_type = config.optimizer_type
        activation_type = config.activation_type
        hidden_dim = config.hidden_dim
        hidden_layers = config.hidden_layers

        model = MLP_classifier(
            input_dim=X_train_processed.shape[1], 
            output_dim=output_dim,
            n_hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim,
            lr=l_rate, 
            max_iter=5000,
            optimizer_type=optimizer_type,
            act_type=activation_type
        )

        # Convert to numpy arrays if necessary
        X_train_array = X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val_processed.values if hasattr(X_val_processed, 'values') else X_val_processed
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        X_test_array = X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        model.fit(X_train_array, y_train_array, X_val_array, y_val_array, wandb_log=True)
        a_s, p_s, r_s, f_s = model.evaluate(X_val_array, X_test_array, y_val_array, y_test_array)
        
        results.append({
            'lr': l_rate,
            'optimizer_type': optimizer_type,
            'activation_type': activation_type,
            'hidden_dim': hidden_dim,
            'hidden_layers': hidden_layers,
            'validation/accuracy': a_s,
            'validaton/precision': p_s,
            'validation/recall': r_s,
            'validation/f1_score': f_s
        })
        wandb.log({
            "lr": l_rate,
            "optimizer_type": optimizer_type,
            "activation_type": activation_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            'validation/accuracy': a_s,
            'validaton/precision': p_s,
            'validation/recall': r_s,
            'validation/f1_score': f_s

        })

    wandb.agent(sweep_id, function=tune_model)
    df = pd.DataFrame(results)
    df = df.sort_values(by="validation/accuracy", ascending=False)
    df = df.reset_index(drop=True)
    print(f"Hyperparameter Tuning Results")
    print(df.to_string(index=False))
    best_params = df.iloc[0].to_dict()
    print(f"\bBest Model Parameters")
    print(best_params)
    wandb.finish()

def task_2_4(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False):
    wandb_log = wandb_log
    L_RATE = 5e-4
    MAX_ITERATIONS = 10000
    OPTIMIZER_TYPE = "mini"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=1


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS, 
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True
    )

    # Print the shapes of each weights (to debug)
    for i in range(NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    NN_model.save_model("mini_sigmoid_20_1.pkl")
    if wandb_log:
        wandb.finish()

    load_and_evaluate_model("mini_sigmoid_20_1.pkl", X_val_processed, y_val, X_test_processed, y_test)
    
def task_2_5_1(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False):
    l_rate_list = [1e-3]
    optimizer_type_list = ["mini"]
    activation_type_list = ["sigmoid", "tanh", "relu", "leakyrelu"]
    hidden_dim_list = [20]
    hidden_layers_list = [1]

    wandb.init(
        project="MLP_Classification"
    )

    results = []

    sweep_config = {
        "method": "grid",
        "name": "Effect_of_non_linearity",
        "metric": {
            "goal": "maximize",
            "name": "test/accuracy"
        },
        "parameters": {
            "l_rate": {
                "values": l_rate_list
            },
            "optimizer_type": {
                "values": optimizer_type_list
            },
            "activation_type": {
                "values": activation_type_list
            },
            "hidden_dim": {
                "values": hidden_dim_list
            },
            "hidden_layers": {
                "values": hidden_layers_list
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project = "MLP_Classification"
    )

    def tune_model():
        wandb.init(
            project = "MLP_Classification"
        )
        config = wandb.config
        l_rate = config.l_rate
        optimizer_type = config.optimizer_type
        activation_type = config.activation_type
        hidden_dim = config.hidden_dim
        hidden_layers = config.hidden_layers

        model = MLP_classifier(
            input_dim=X_train_processed.shape[1], 
            output_dim=output_dim,
            n_hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim,
            lr=l_rate, 
            max_iter=10000,
            optimizer_type=optimizer_type,
            act_type=activation_type
        )

        # Convert to numpy arrays if necessary
        X_train_array = X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val_processed.values if hasattr(X_val_processed, 'values') else X_val_processed
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        X_test_array = X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        model.fit(X_train_array, y_train_array, X_val_array, y_val_array, wandb_log=True)
        a_s, p_s, r_s, f_s = model.evaluate(X_val_array, X_test_array, y_val_array, y_test_array)
        
        results.append({
            'lr': l_rate,
            'optimizer_type': optimizer_type,
            'activation_type': activation_type,
            'hidden_dim': hidden_dim,
            'hidden_layers': hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s
        })
        wandb.log({
            "lr": l_rate,
            "optimizer_type": optimizer_type,
            "activation_type": activation_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s

        })

    wandb.agent(sweep_id, function=tune_model)
    df = pd.DataFrame(results)
    df = df.sort_values(by="test/accuracy", ascending=False)
    df = df.reset_index(drop=True)
    print(f"Hyperparameter Tuning Results")
    print(df.to_string(index=False))
    best_params = df.iloc[0].to_dict()
    print(f"\bBest Model Parameters")
    print(best_params)
    wandb.finish()

    """
    Hyperparameter Tuning Results
    lr optimizer_type activation_type  hidden_dim  hidden_layers  test/accuracy  test/precision  test/recall  test/f1_score
    0.001           mini         sigmoid          20              1       0.614035        0.275519     0.269925       0.614035
    0.001           mini            tanh          20              1       0.587719        0.256469     0.285505       0.587719
    0.001           mini            relu          20              1       0.543860        0.239881     0.256099       0.543860
    0.001           mini       leakyrelu          20              1       0.543860        0.241219     0.267753       0.543860
    Best Model Parameters
    {'lr': 0.001, 'optimizer_type': 'mini', 'activation_type': 'sigmoid', 'hidden_dim': 20, 'hidden_layers': 1, 'test/accuracy': 0.6140350877192983, 'test/precision': 0.27551892551892554, 'test/recall': 0.2699248120300752, 'test/f1_score': 0.6140350877192983}
    """

def task_2_5_2(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False):
    l_rate_list = [1e-2, 1e-3, 5e-3, 1e-4]
    optimizer_type_list = ["mini"]
    activation_type_list = ["sigmoid"]
    hidden_dim_list = [20]
    hidden_layers_list = [1]

    wandb.init(
        project="MLP_Classification"
    )

    results = []

    sweep_config = {
        "method": "grid",
        "name": "Effect_of_learning_rate",
        "metric": {
            "goal": "maximize",
            "name": "test/accuracy"
        },
        "parameters": {
            "l_rate": {
                "values": l_rate_list
            },
            "optimizer_type": {
                "values": optimizer_type_list
            },
            "activation_type": {
                "values": activation_type_list
            },
            "hidden_dim": {
                "values": hidden_dim_list
            },
            "hidden_layers": {
                "values": hidden_layers_list
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project = "MLP_Classification"
    )

    def tune_model():
        wandb.init(
            project = "MLP_Classification"
        )
        config = wandb.config
        l_rate = config.l_rate
        optimizer_type = config.optimizer_type
        activation_type = config.activation_type
        hidden_dim = config.hidden_dim
        hidden_layers = config.hidden_layers

        model = MLP_classifier(
            input_dim=X_train_processed.shape[1], 
            output_dim=output_dim,
            n_hidden_layers=hidden_layers, 
            hidden_dim=hidden_dim,
            lr=l_rate, 
            max_iter=10000,
            optimizer_type=optimizer_type,
            act_type=activation_type
        )

        # Convert to numpy arrays if necessary
        X_train_array = X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val_processed.values if hasattr(X_val_processed, 'values') else X_val_processed
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        X_test_array = X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        model.fit(X_train_array, y_train_array, X_val_array, y_val_array, wandb_log=True, batch_size=config.batch_size)
        a_s, p_s, r_s, f_s = model.evaluate(X_val_array, X_test_array, y_val_array, y_test_array)
        
        results.append({
            'lr': l_rate,
            'optimizer_type': optimizer_type,
            'activation_type': activation_type,
            'hidden_dim': hidden_dim,
            'hidden_layers': hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s
        })
        wandb.log({
            "lr": l_rate,
            "optimizer_type": optimizer_type,
            "activation_type": activation_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s

        })

    wandb.agent(sweep_id, function=tune_model)
    df = pd.DataFrame(results)
    df = df.sort_values(by="test/accuracy", ascending=False)
    df = df.reset_index(drop=True)
    print(f"Hyperparameter Tuning Results")
    print(df.to_string(index=False))
    best_params = df.iloc[0].to_dict()
    print(f"\bBest Model Parameters")
    print(best_params)
    wandb.finish()

    """
    Hyperparameter Tuning Results
        lr optimizer_type activation_type  hidden_dim  hidden_layers  test/accuracy  test/precision  test/recall  test/f1_score
    0.0010           mini         sigmoid          20              1       0.649123        0.300098     0.298496       0.649123
    0.0050           mini         sigmoid          20              1       0.614035        0.269180     0.295322       0.614035
    0.0001           mini         sigmoid          20              1       0.578947        0.187762     0.215957       0.578947
    0.0100           mini         sigmoid          20              1       0.543860        0.238360     0.267753       0.543860
    Best Model Parameters
    {'lr': 0.001, 'optimizer_type': 'mini', 'activation_type': 'sigmoid', 'hidden_dim': 20, 'hidden_layers': 1, 'test/accuracy': 0.6491228070175439, 'test/precision': 0.30009775171065495, 'test/recall': 0.29849624060150376, 'test/f1_score': 0.6491228070175439}
    """

def task_2_5_3(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False):
    l_rate_list = [1e-3]
    optimizer_type_list = ["mini"]
    activation_type_list = ["sigmoid"]
    hidden_dim_list = [20]
    hidden_layers_list = [1]
    batch_sizes = [8, 16, 32, 64]

    wandb.init(
        project="MLP_Classification"
    )

    results = []

    sweep_config = {
        "method": "grid",
        "name": "Effect_of_batch_size",
        "metric": {
            "goal": "maximize",
            "name": "test/accuracy"
        },
        "parameters": {
            "l_rate": {
                "values": l_rate_list
            },
            "optimizer_type": {
                "values": optimizer_type_list
            },
            "activation_type": {
                "values": activation_type_list
            },
            "hidden_dim": {
                "values": hidden_dim_list
            },
            "hidden_layers": {
                "values": hidden_layers_list
            },
            "batch_size": {
                "values": batch_sizes
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project = "MLP_Classification"
    )

    def tune_model():
        wandb.init(
            project = "MLP_Classification"
        )
        config = wandb.config
        l_rate = config.l_rate
        optimizer_type = config.optimizer_type
        activation_type = config.activation_type
        hidden_dim = config.hidden_dim
        hidden_layers = config.hidden_layers
        batch_size = config.batch_size

        model = MLP_classifier(
            input_dim=X_train_processed.shape[1], 
            output_dim=output_dim,
            n_hidden_layers=hidden_layers,
            hidden_dim=hidden_dim,
            lr=l_rate,
            max_iter=10000,
            optimizer_type=optimizer_type,
            act_type=activation_type
        )

        # Convert to numpy arrays if necessary
        X_train_array = X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val_processed.values if hasattr(X_val_processed, 'values') else X_val_processed
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        X_test_array = X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        model.fit(X_train_array, y_train_array, X_val_array, y_val_array, wandb_log=True, batch_size=batch_size)
        a_s, p_s, r_s, f_s = model.evaluate(X_val_array, X_test_array, y_val_array, y_test_array)
        
        results.append({
            'lr': l_rate,
            'optimizer_type': optimizer_type,
            'activation_type': activation_type,
            'hidden_dim': hidden_dim,
            'hidden_layers': hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s
        })
        wandb.log({
            "lr": l_rate,
            "optimizer_type": optimizer_type,
            "activation_type": activation_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            'test/accuracy': a_s,
            'test/precision': p_s,
            'test/recall': r_s,
            'test/f1_score': f_s
        })

    wandb.agent(sweep_id, function=tune_model)
    df = pd.DataFrame(results)
    df = df.sort_values(by="test/accuracy", ascending=False)
    df = df.reset_index(drop=True)
    print(f"Hyperparameter Tuning Results")
    print(df.to_string(index=False))
    best_params = df.iloc[0].to_dict()
    print(f"\bBest Model Parameters")
    print(best_params)
    wandb.finish()

    """
    Hyperparameter Tuning Results
    lr optimizer_type activation_type  hidden_dim  hidden_layers  test/accuracy  test/precision  test/recall  test/f1_score
    0.001           mini         sigmoid          20              1       0.675439        0.358997     0.324144       0.675439
    0.001           mini         sigmoid          20              1       0.614035        0.268675     0.282623       0.614035
    0.001           mini         sigmoid          20              1       0.596491        0.255955     0.274687       0.596491
    0.001           mini         sigmoid          20              1       0.596491        0.270581     0.265121       0.596491
    Best Model Parameters
    {'lr': 0.001, 'optimizer_type': 'mini', 'activation_type': 'sigmoid', 'hidden_dim': 20, 'hidden_layers': 1, 'test/accuracy': 0.6754385964912281, 'test/precision': 0.3589972712248741, 'test/recall': 0.3241436925647452, 'test/f1_score': 0.6754385964912281}
    """

def task_2_6(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=False, do_multi_label:bool=True):
    wandb_log = wandb_log
    L_RATE = 1e-3
    MAX_ITERATIONS = 10000
    OPTIMIZER_TYPE = "mini"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=1


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS, 
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True,
        do_multi_label=do_multi_label
    )



    # Print the shapes of each weights (to debug)
    for i in range(NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)


def task_3_2(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim:int, wandb_log=False):
    wandb_log = wandb_log
    L_RATE = 1e-3
    MAX_ITERATIONS = 500
    OPTIMIZER_TYPE = "sgd"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=2


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True,
        objective="regression"
    )



    # Print the shapes of each weights (to debug)
    for i in range(0, NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)

def task_3_3(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim: int, wandb_log=True):
    l_rate_list = [1e-3, 5e-4, 1e-4]
    optimizer_type_list = ["batch", "mini", "sgd"]
    activation_type_list = ["linear", "sigmoid", "tanh", "relu"]
    hidden_dim_list = [11, 15, 20]
    hidden_layers_list = [1, 2]

    results = []

    sweep_config = {
        "method": "grid",
        "name": "MLP_Regression_Hyperparameter_Tuning",
        "metric": {
            "goal": "minimize",
            "name": "validation/mse"
        },
        "parameters": {
            "l_rate": {
                "values": l_rate_list
            },
            "optimizer_type": {
                "values": optimizer_type_list
            },
            "activation_type": {
                "values": activation_type_list
            },
            "hidden_dim": {
                "values": hidden_dim_list
            },
            "hidden_layers": {
                "values": hidden_layers_list
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config,
        project = "SMAI-A3-MLP"
    )

    def tune_model():
        wandb.init(
            project = "SMAI-A3-MLP"
        )
        config = wandb.config
        l_rate = config.l_rate
        optimizer_type = config.optimizer_type
        activation_type = config.activation_type
        hidden_dim = config.hidden_dim
        hidden_layers = config.hidden_layers
        
        if optimizer_type == "sgd":
            model = MLP_classifier(
                input_dim=X_train_processed.shape[1], 
                output_dim=output_dim,
                n_hidden_layers=hidden_layers, 
                hidden_dim=hidden_dim,
                lr=l_rate, 
                max_iter=500,
                optimizer_type=optimizer_type,
                act_type=activation_type,
                objective="regression"
            )
        else:
            model = MLP_classifier(
                input_dim=X_train_processed.shape[1], 
                output_dim=output_dim,
                n_hidden_layers=hidden_layers, 
                hidden_dim=hidden_dim,
                lr=l_rate,
                max_iter=9000,
                optimizer_type=optimizer_type,
                act_type=activation_type,
                objective="regression"
            )
        # Convert to numpy arrays if necessary
        X_train_array = X_train_processed.values if hasattr(X_train_processed, 'values') else X_train_processed
        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_array = X_val_processed.values if hasattr(X_val_processed, 'values') else X_val_processed
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        X_test_array = X_test_processed.values if hasattr(X_test_processed, 'values') else X_test_processed
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        model.fit(X_train_array, y_train_array, X_val_array, y_val_array, wandb_log=True)
        mse, rmse, r_squared = model.evaluate(X_val_array, X_test_array, y_val_array, y_test_array)
        
        results.append({
            'lr': l_rate,
            'optimizer_type': optimizer_type,
            'activation_type': activation_type,
            'hidden_dim': hidden_dim,
            'hidden_layers': hidden_layers,
            'validation/mse': mse,
            'validation/rmse': rmse,
            'validation/r_squared': r_squared
        })
        wandb.log({
            "lr": l_rate,
            "optimizer_type": optimizer_type,
            "activation_type": activation_type,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            'validation/mse': mse,
            'validation/rmse': rmse,
            'validation/r_squared': r_squared
        })

    wandb.agent(sweep_id, function=tune_model)
    df = pd.DataFrame(results)
    df = df.sort_values(by="validation/mse", ascending=True)
    df = df.reset_index(drop=True)
    print(f"Hyperparameter Tuning Results")
    print(df.to_string(index=False))
    best_params = df.iloc[0].to_dict()
    print(f"\bBest Model Parameters")
    print(best_params)
    wandb.finish()

def task_3_4(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim:int, wandb_log=False):
    wandb_log = wandb_log
    L_RATE = 1e-3
    MAX_ITERATIONS = 500
    OPTIMIZER_TYPE = "sgd"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=2


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True,
        objective="regression"
    )



    # Print the shapes of each weights (to debug)
    for i in range(0, NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)

def remove_outliers(df, columns, n_std=3):
    """
    Remove outliers using the IQR method
    Parameters:
    df : DataFrame
    columns : list of column names to check for outliers
    n_std : number of standard deviations to use for the IQR method
    
    Returns:
    DataFrame with outliers removed
    """
    df_clean = df.copy()
    for column in columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - n_std * IQR
        upper_bound = Q3 + n_std * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean


def task_4_3():
    LATENT_DIM = 3
    HIDDEN_LAYERS = 1
    HIDDEN_DIM = 20
    L_RATE = 1e-3
    act_type = "sigmoid"
    optimizer_type = "mini"
    BATCH_SIZE = 32
    MAX_ITER = 50

    # read the spotify dataset
    df = pd.read_csv("spotify.csv")
    df.dropna(inplace=True)
    print(f"Dataset info: {df.info()}")
    features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo']
    
    normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())

    # Create a new column 'vit' which is a list of all other column values for each row
    normalized_df['vit'] = normalized_df.apply(lambda row: row.tolist(), axis=1)

    X = np.array(normalized_df['vit'].tolist())
    
    print(f"X_shape: {X.shape}")

    AutoEncoder_model =  AutoEncoder(X.shape[1], LATENT_DIM, HIDDEN_LAYERS, HIDDEN_DIM, L_RATE, act_type=act_type, optimizer_type=optimizer_type, batch_size=BATCH_SIZE, max_iter=MAX_ITER)
    AutoEncoder_model.fit(X, wandb_log=False, max_iterations=MAX_ITER)

    reduced_dataset = AutoEncoder_model.get_latent(X)
    print(f"reduced_dataset: {reduced_dataset}")
    
    pca_df = pd.DataFrame(reduced_dataset, columns=['col1', 'col2', 'col3'])
    pca_df['track_genre'] = df['track_genre']
    print(pca_df.head())


    features_list = ['col1', 'col2', 'col3']
    k = 15
    metric = 'cosine'
    model = Best_KNN_model(k, metric, features=features_list)
    model.train(pca_df)
    model.split_data(validation_split=0.1, test_split=0.1)
    evaluator = Best_KNN_evaluate(model)
    validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid.astype(str))
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
    # print(results)
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    df_print = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
    print(df_print.to_string(index=False))


def task_4_4():
    LATENT_DIM = 3
    HIDDEN_LAYERS = 1
    HIDDEN_DIM = 15
    L_RATE = 1e-3
    act_type = "sigmoid"
    optimizer_type = "mini"
    BATCH_SIZE = 32
    MAX_ITER = 100

    df = pd.read_csv("spotify.csv")
    df.dropna(inplace=True)
    df = df[:20000]

    print(f"Dataset info: {df.info()}")
    features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'tempo']
    
    normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())

    # Create a new column 'vit' which is a list of all other column values for each row
    normalized_df['vit'] = normalized_df.apply(lambda row: row.tolist(), axis=1)

    X = np.array(normalized_df['vit'].tolist())
    
    print(f"X_shape: {X.shape}")

    AutoEncoder_model =  AutoEncoder(X.shape[1], LATENT_DIM, HIDDEN_LAYERS, HIDDEN_DIM, L_RATE, act_type=act_type, optimizer_type=optimizer_type, batch_size=BATCH_SIZE, max_iter=MAX_ITER)
    AutoEncoder_model.fit(X, wandb_log=False, max_iterations=MAX_ITER)

    reduced_dataset = AutoEncoder_model.get_latent(X)
    print(f"reduced_dataset: {reduced_dataset}")
    
    pca_df = pd.DataFrame(reduced_dataset, columns=['col1', 'col2', 'col3'])
    pca_df['track_genre'] = df['track_genre']
    print(pca_df.head())


    # MLP Classifier
    df_copy = pca_df

    X = df_copy.drop('track_genre', axis=1)
    y = df_copy['track_genre']
    y = y.astype(str)
    print(y)

    unique_output_vals = len(np.unique(y))
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # If you have only 2 classes, LabelBinarizer will return a 1D array
    # To make it consistent with multi-class case, reshape it:
    if len(lb.classes_) == 2:
        y = np.hstack((1 - y, y))
    
    print(y)

    # Split
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.33, random_state=42, shuffle=True)

    # Print n_samples in each
    print(f"num_samples in X_train: {len(X_train)}")
    print(f"num_samples in X_val: {len(X_val)}")
    print(f"num_samples in X_test: {len(X_test)}")

    # Intialize the scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # # Standardize the normalized data
    X_train_standardized = standard_scaler.fit_transform(X_train)
    X_test_standardized = standard_scaler.fit_transform(X_test)
    X_val_standardized = standard_scaler.fit_transform(X_val)

    # Normalize the data using Min-Max scaling

    # On normal data
    X_train_normalized = min_max_scaler.fit_transform(X_train)
    X_val_normalized = min_max_scaler.fit_transform(X_val)
    X_test_normalized = min_max_scaler.fit_transform(X_test)

    # Normalize standardized data
    # X_train_normalized = min_max_scaler.fit_transform(X_train_standardized)
    # X_val_normalized = min_max_scaler.fit_transform(X_val_standardized)
    # X_test_normalized = min_max_scaler.fit_transform(X_test_standardized)

    # Convert the scaled arrays back to DataFrames
    X_train_processed = pd.DataFrame(X_train_normalized, columns=X.columns)
    X_test_processed = pd.DataFrame(X_test_normalized, columns=X.columns)
    X_val_processed = pd.DataFrame(X_val_normalized, columns=X.columns)

    # X_train_processed = pd.DataFrame(X_train_standardized, columns=X.columns)
    # X_test_processed = pd.DataFrame(X_test_standardized, columns=X.columns)
    # X_val_processed = pd.DataFrame(X_val_standardized, columns=X.columns)

    # Function to calculate and print statistics
    def print_stats(data, name):
        # print(f"\n{name} (first 5 rows):")
        # print(data.head())
        print(f"\n{name} statistics:")
        print(data.describe().loc[['mean', 'std', 'min', 'max']].T)

    # Print statistics for original and processed data
    print_stats(X_train, "Original data")

    # Verify the range of original data
    print("\nRange of original data:")
    print(f"Min: {X_train.min().min()}")
    print(f"Max: {X_train.max().max()}")

    print_stats(X_train_processed, "Normalized data")

    # Verify the range of processed data
    print("\nRange of processed data:")
    print(f"Min: {X_train_processed.min().min()}")
    print(f"Max: {X_train_processed.max().max()}")

    L_RATE = 1e-3
    MAX_ITERATIONS = 1000
    OPTIMIZER_TYPE = "mini"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM = 20
    HIDDEN_LAYERS = 1
    wandb_log = False
    output_dim = unique_output_vals

    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS, 
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True
    )



    # Print the shapes of each weights (to debug)
    for i in range(NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)


def task_3_5(X_train_processed: np.ndarray, X_val_processed: np.ndarray, X_test_processed: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, output_dim:int, wandb_log=False):
    wandb_log = wandb_log
    L_RATE = 1e-3
    MAX_ITERATIONS = 500
    OPTIMIZER_TYPE = "mini"
    ACTIVATION_TYPE = "sigmoid"
    HIDDEN_DIM=20
    HIDDEN_LAYERS=2


    np.random.seed(42)
    NN_model = MLP_classifier(
        input_dim=X_train_processed.shape[1], 
        output_dim=output_dim,
        n_hidden_layers=HIDDEN_LAYERS,
        hidden_dim=HIDDEN_DIM,
        lr=L_RATE, 
        max_iter=MAX_ITERATIONS,
        optimizer_type=OPTIMIZER_TYPE,
        act_type=ACTIVATION_TYPE,
        do_gradient_check=True,
        objective="regression"
    )



    # Print the shapes of each weights (to debug)
    for i in range(0, NN_model.n_hidden_layers + 1):
        print(f"{i}th weight shape: {NN_model.weights[i].shape}")
        
    if wandb_log: 
        wandb.login()
        wandb.init(
            project = "SMAI-A3",
            name = f"MLP_classifier",
            entity="vemulasakethreddy_10",
            config={
                "lr": L_RATE,
                "max_iterations": MAX_ITERATIONS,
                "optimizer_type": OPTIMIZER_TYPE,
                "activation_function": ACTIVATION_TYPE,
                "hidden_dim": HIDDEN_DIM,
                "hidden_layers": HIDDEN_LAYERS,
                "batch_size": 32
            }
        )
    # Training
    NN_model.fit(np.array(X_train_processed), np.array(y_train), np.array(X_val_processed), np.array(y_val), wandb_log=wandb_log)
    if wandb_log:
        wandb.finish()
    
    NN_model.evaluate(X_train_processed, X_test_processed, y_train, y_test)


if __name__ == "__main__":
    do_preprocessing_visualization = False
    single_label_classification = False
    multi_label_classification = False
    regression = False
    autoencoder = False
    binary_classification = True

    if do_preprocessing_visualization == True:
        task_2_1()
    elif single_label_classification:
        # read the data
        df = pd.read_csv("WineQT.csv")
        df.dropna()
        df = df.drop(['Id'], axis=1)
        df_copy = df


        X = df_copy.drop('quality', axis=1)
        y = df_copy['quality']
        unique_output_vals = len(np.unique(y))
        print(y)
        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        # If you have only 2 classes, LabelBinarizer will return a 1D array
        # To make it consistent with multi-class case, reshape it:
        if len(lb.classes_) == 2:
            y = np.hstack((1 - y, y))
        
        print(y)

        # Split
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.33, random_state=42, shuffle=True)

        # Print n_samples in each
        print(f"num_samples in X_train: {len(X_train)}")
        print(f"num_samples in X_val: {len(X_val)}")
        print(f"num_samples in X_test: {len(X_test)}")

        # Intialize the scalers
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # # Standardize the normalized data
        X_train_standardized = standard_scaler.fit_transform(X_train)
        X_test_standardized = standard_scaler.fit_transform(X_test)
        X_val_standardized = standard_scaler.fit_transform(X_val)

        # Normalize the data using Min-Max scaling

        # On normal data
        X_train_normalized = min_max_scaler.fit_transform(X_train)
        X_val_normalized = min_max_scaler.fit_transform(X_val)
        X_test_normalized = min_max_scaler.fit_transform(X_test)

        # Normalize standardized data
        # X_train_normalized = min_max_scaler.fit_transform(X_train_standardized)
        # X_val_normalized = min_max_scaler.fit_transform(X_val_standardized)
        # X_test_normalized = min_max_scaler.fit_transform(X_test_standardized)

        # Convert the scaled arrays back to DataFrames
        X_train_processed = pd.DataFrame(X_train_normalized, columns=X.columns)
        X_test_processed = pd.DataFrame(X_test_normalized, columns=X.columns)
        X_val_processed = pd.DataFrame(X_val_normalized, columns=X.columns)

        # X_train_processed = pd.DataFrame(X_train_standardized, columns=X.columns)
        # X_test_processed = pd.DataFrame(X_test_standardized, columns=X.columns)
        # X_val_processed = pd.DataFrame(X_val_standardized, columns=X.columns)

        # Function to calculate and print statistics
        def print_stats(data, name):
            # print(f"\n{name} (first 5 rows):")
            # print(data.head())
            print(f"\n{name} statistics:")
            print(data.describe().loc[['mean', 'std', 'min', 'max']].T)

        # Print statistics for original and processed data
        print_stats(X_train, "Original data")

        # Verify the range of original data
        print("\nRange of original data:")
        print(f"Min: {X_train.min().min()}")
        print(f"Max: {X_train.max().max()}")

        print_stats(X_train_processed, "Normalized data")

        # Verify the range of processed data
        print("\nRange of processed data:")
        print(f"Min: {X_train_processed.min().min()}")
        print(f"Max: {X_train_processed.max().max()}")

        df.hist(bins=20, figsize=(15, 10))
        plt.savefig("figures/WineQT.png")


        # task_2_2(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False, output_dim=unique_output_vals) # Model building from Scratch (Gradient Checking)
        # task_2_3(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=True, output_dim=unique_output_vals) # Model Training and Hyperparameter Tuning
        task_2_4(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=True, output_dim=unique_output_vals) # Evaluating Single-label classification Model

        # task_2_5_1(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False, output_dim=unique_output_vals) # Effect of Non-linearity
        # task_2_5_2(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False, output_dim=unique_output_vals) # Effect of learning rate
        # task_2_5_3(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False, output_dim=unique_output_vals) # Effect of batch size

    elif multi_label_classification:
        df = pd.read_csv("advertisement.csv")
        # print(df.head())
        df.dropna()
        # print(df.info())

        df.dropna(inplace=True)
        numerical_features = ['age', 'income', 'children', 'purchase_amount']
        categorical_features = ['gender', 'education', 'married', 'city', 'occupation', 'most bought item']
        target_column = ['labels']

        # Encoder Catagorical Variables
        """
            This performs one-hot encoding on all categorical features:
            - Creates binary columns for each category in each categorical feature
            - 'drop_first=True' removes one category from each feature to avoid multicollinearity
            Example:
            Before: gender = ['M', 'F']
            After: gender_M = [1, 0], gender_F = [0, 1] (but gender_F is dropped)
        """
        encodings = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        X = encodings.drop('labels', axis=1)

        # print(f"X: {X.columns}")
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # print(f"Shape: {X.shape}")
        # print(f"X: {X}")

        y = encodings['labels']
        mlb = MultiLabelBinarizer()
        y_binary = (mlb.fit_transform(y.str.split(' ')))

        y_new = []
        for i in range(y_binary.shape[0]):
            if np.sum(y_binary[i]) > 0:
                y_new.append(y_binary[i]/np.sum(y_binary[i]))
        y_binary = np.array(y_new)
        unique_output_vals = len(y_binary[0])

        X_train, X_rem, y_train, y_rem = train_test_split(X, y_binary, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.67, random_state=42, shuffle=True)

        indices = (y_val > 0)
        y_val[indices] = 1
        indices = (y_test > 0)
        y_test[indices] = 1

        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(X_test.shape)
        print(y_test.shape)

        # print(X_val)

        # Print n_samples in each
        print(f"num_samples in X_train: {len(X_train)}")
        print(f"num_samples in X_val: {len(X_val)}")
        print(f"num_samples in X_test: {len(X_test)}")
        
        print(y_test)

        task_2_6(X_train, X_val, X_test, y_train, y_val, y_test, wandb_log=False, output_dim=unique_output_vals, do_multi_label=True)

    elif regression == True:
        do_remove_outliers = True
        # read the data
        df = pd.read_csv("HousingData.csv")
        df.dropna(inplace=True)
        # print(df.head(20))
        print(f"Dataset Info: {df.info()}")

        df_copy = df

        print(f"Dataset shape before removing outliers: {df.shape}")

        df.hist(bins=20, figsize=(15, 10))
        plt.savefig("figures/HousingData_before_outlier_removal.png")

        # Remove outliers from all numeric columns except the target variable
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove('MEDV')  # Remove target variable from outlier removal

        df_no_outliers = remove_outliers(df, numeric_columns)
        print(f"Dataset shape after removing outliers: {df_no_outliers.shape}")

        df_no_outliers.hist(bins=20, figsize=(15, 10))
        plt.savefig("figures/HousingData_after_outlier_removal.png")

        if do_remove_outliers:
            df_copy = df_no_outliers


        X = df_copy.drop('MEDV', axis=1)
        y = df_copy['MEDV']
        unique_output_vals = len(np.unique(y))
        # print(y)
    

        # Split
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.67, random_state=42, shuffle=True)

        # Print n_samples in each
        print(f"num_samples in X_train: {len(X_train)}")
        print(f"num_samples in X_val: {len(X_val)}")
        print(f"num_samples in X_test: {len(X_test)}")

        # Intialize the scalers
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Standardize the data
        X_train_standardized = standard_scaler.fit_transform(X_train)
        X_test_standardized = standard_scaler.fit_transform(X_test)
        X_val_standardized = standard_scaler.fit_transform(X_val)

        # Normalize the data using Min-Max scaling

        # On normal data
        X_train_normalized = min_max_scaler.fit_transform(X_train)
        X_val_normalized = min_max_scaler.fit_transform(X_val)
        X_test_normalized = min_max_scaler.fit_transform(X_test)

        # Standardize normalized data
        # X_train_normalized = standard_scaler.fit_transform(X_train_normalized)
        # X_val_normalized = standard_scaler.fit_transform(X_val_normalized)
        # X_test_normalized = standard_scaler.fit_transform(X_test_normalized)
        
        # Normalize standardized data
        X_train_normalized = min_max_scaler.fit_transform(X_train_standardized)
        X_val_normalized = min_max_scaler.fit_transform(X_val_standardized)
        X_test_normalized = min_max_scaler.fit_transform(X_test_standardized)

        # Convert the scaled arrays back to DataFrames
        X_train_processed = pd.DataFrame(X_train_normalized, columns=X.columns)
        X_test_processed = pd.DataFrame(X_test_normalized, columns=X.columns)
        X_val_processed = pd.DataFrame(X_val_normalized, columns=X.columns)

        # X_train_processed = pd.DataFrame(X_train_standardized, columns=X.columns)
        # X_test_processed = pd.DataFrame(X_test_standardized, columns=X.columns)
        # X_val_processed = pd.DataFrame(X_val_standardized, columns=X.columns)

        # Function to calculate and print statistics
        def print_stats(data, name):
            # print(f"\n{name} (first 5 rows):")
            # print(data.head())
            print(f"\n{name} statistics:")
            print(data.describe().loc[['mean', 'std', 'min', 'max']].T)

        # Print statistics for original and processed data
        print_stats(X_train, "Original data")

        # Verify the range of original data
        print("\nRange of original data:")
        print(f"Min: {X_train.min().min():.4f}")
        print(f"Max: {X_train.max().max():.4f}")

        print_stats(X_train_processed, "Normalized data")

        # Verify the range of processed data
        print("\nRange of processed data:")
        print(f"Min: {X_train_processed.min().min():.4f}")
        print(f"Max: {X_train_processed.max().max():.4f}")
        # if self.objective == "regression":
        #     y = y.reshape(-1, 1)
        #     y_val = y_val.reshape(-1, 1)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print(f"y_train-shape: {y_train.shape}")
        print(f"y_val_shape: {y_val.shape}")
        print(f"y_test_shape: {y_test.shape}")

        # task_3_2(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, output_dim=1, wandb_log=False) # Model building from Scratch (Gradient Checking)
        # task_3_3(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, output_dim=1, wandb_log=True) # Model Training and Hyperparameter Tuning
        # task_3_5_1(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False) # Effect of Non-linearity
        # task_3_5_2(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False) # Effect of learning rate
        # task_3_5_3(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, wandb_log=False) # Effect of batch size

    elif autoencoder == True:
        
        # task_4_3()
        task_4_4()

    elif binary_classification == True:
        df = pd.read_csv("diabetes.csv")
        df.dropna(inplace=True)
        print(df.head())
        df_copy = df
        X = df_copy.drop('Outcome', axis=1)
        y = df_copy['Outcome']

        unique_output_vals = len(np.unique(y))
        print(y)

        lb = LabelBinarizer()
        y = lb.fit_transform(y)

        if len(lb.classes_) == 2:
            y = np.hstack((1 - y, y))

        print(y)

        # Split
        X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.67, random_state=42, shuffle=True)

        #print n_samples in each
        print(f"num_samples in X_train: {len(X_train)}")
        print(f"num_samples in X_val: {len(X_val)}")
        print(f"num_samples in X_test: {len(X_test)}")

        # Intialize the scalers
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Standardize the normalized data
        X_train_standardized = standard_scaler.fit_transform(X_train)
        X_test_standardized = standard_scaler.fit_transform(X_test)
        X_val_standardized = standard_scaler.fit_transform(X_val)

        # Normalize the data using Min-Max scaling

        # On normal data
        X_train_normalized = min_max_scaler.fit_transform(X_train)
        X_val_normalized = min_max_scaler.fit_transform(X_val)
        X_test_normalized = min_max_scaler.fit_transform(X_test)

        # Normalize standardized data
        # X_train_normalized = min_max_scaler.fit_transform(X_train_standardized)
        # X_val_normalized = min_max_scaler.fit_transform(X_val_standardized)
        # X_test_normalized = min_max_scaler.fit_transform(X_test_standardized)

        # Convert the scaled arrays back to DataFrames
        X_train_processed = pd.DataFrame(X_train_normalized, columns=X.columns)
        X_test_processed = pd.DataFrame(X_test_normalized, columns=X.columns)
        X_val_processed = pd.DataFrame(X_val_normalized, columns=X.columns)

        # X_train_processed = pd.DataFrame(X_train_standardized, columns=X.columns)
        # X_test_processed = pd.DataFrame(X_test_standardized, columns=X.columns)
        # X_val_processed = pd.DataFrame(X_val_standardized, columns=X.columns)

        # Function to calculate and print statistics
        def print_stats(data, name):
            # print(f"\n{name} (first 5 rows):")
            # print(data.head())
            print(f"\n{name} statistics:")
            print(data.describe().loc[['mean', 'std', 'min', 'max']].T)

        # Print statistics for original and processed data
        print_stats(X_train, "Original data")

        # Verify the range of original data
        print("\nRange of original data:")
        print(f"Min: {X_train.min().min()}")
        print(f"Max: {X_train.max().max()}")

        print_stats(X_train_processed, "Normalized data")

        # Verify the range of processed data
        print("\nRange of processed data:")
        print(f"Min: {X_train_processed.min().min()}")
        print(f"Max: {X_train_processed.max().max()}")

        # if self.objective == "regression":
        #     y = y.reshape(-1, 1)
        #     y_val = y_val.reshape(-1, 1)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        print(f"y_train-shape: {y_train.shape}")
        print(f"y_val_shape: {y_val.shape}")
        print(f"y_test_shape: {y_test.shape}")

        task_3_5(X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, output_dim=1, wandb_log=False)
