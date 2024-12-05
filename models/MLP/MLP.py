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