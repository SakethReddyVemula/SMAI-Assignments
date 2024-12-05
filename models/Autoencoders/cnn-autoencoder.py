import os
import sys
import numpy as np
import pandas as pd
import wandb
import yaml
from omegaconf import OmegaConf
from PIL import Image
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import random, math

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, MultiLabelBinarizer


# Set up wandb key
os.environ["WANDB_API_KEY"] = "c8a7a539cb5fed3df89b21d71956ca6b4befd2a5"

config = OmegaConf.load("../config.yaml")
overrides = OmegaConf.from_cli()
config = OmegaConf.merge(config, overrides)
OmegaConf.to_container(config)
config = OmegaConf.to_container(config, resolve=True)

# set up device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

torch.manual_seed(config["random_state"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(config["random_state"])
    torch.cuda.manual_seed_all(config["random_state"])

def get_one_hot_encoding(label: List[int], num_classes: int=10):
    encoding = torch.zeros(num_classes)
    for digit in label:
        encoding[digit] = 1
    return encoding

def load_and_preprocess_FashionMNIST(root_dir: str):
    original_train = pd.read_csv(root_dir)
    X = original_train.iloc[:, 1:].values.reshape(-1, 28, 28)
    y = original_train['label'].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=config["random_state"])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=config["random_state"])

    return X_train, X_val, X_test, y_train, y_val, y_test

def visualize_one_per_label(X, y):
    plt.figure(figsize=(15, 6))

    unique_labels = np.unique(y)

    for i, label in enumerate(unique_labels):
        idx = np.where(y == label)[0][0]
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx], cmap='gray')
        plt.title(f"Label {label}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("FashionMNIST.png")

    
class FashionMNIST(Dataset):
    def __init__(self, X, y):
        super(FashionMNIST, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        image = torch.from_numpy(np.array(self.X[idx])).float() / 255.0
        image = image.unsqueeze(0)
        label = torch.tensor(get_one_hot_encoding([self.y[idx]], 10))
        # print(f"label: {label}")
        # print(f"label type: {type(label)}")
        # print(f"image size: {image.shape}")
        # print(f"label shape: {label.shape}")
        return image, label


# @params: lr, kernel_size, n_filters in each layer, optimizer
class CNNAutoEncoder(nn.Module):
    def __init__(
            self,
            lr: float=1e-3,
            input_channels: int=1,
            input_size: int=28,
            latent_dim: int=128,
            optimizer: str="AdamW",
            kernel_size: int=3,
            padding: int=1,
            dropout: float=0.1,
            num_classes: int=10,
            classification_hidden_dims: List[int]=[64, 32],
            filters_per_layer: List[int]=[32, 64, 128]
    ):
        super(CNNAutoEncoder, self).__init__()

        current_channels = input_channels
        current_size = input_size
        self.n_cnn_layers = len(filters_per_layer)
        n_cnn_layers = self.n_cnn_layers
        self.filters_per_layer = filters_per_layer
        # ENCODE
        # encoder component
        print(f"Encode Block")
        self.encoder = nn.ModuleList()

        for i in range(n_cnn_layers):
            # out_channels = current_channels * 2 if i > 0 else 32
            out_channels = filters_per_layer[i]
            self.encoder.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            current_size = (current_size + 2 * padding - kernel_size) // 1 + 1
            self.encoder.append(nn.BatchNorm2d(out_channels))
            self.encoder.append(nn.ReLU())
            if current_size // 2 >= 1:
                self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size //= 2
            
            self.encoder.append(nn.Dropout(dropout))
            current_channels = out_channels
            print(f"Encoder: Layer {i+1} output size: {current_size}x{current_size}, channels: {current_channels}")

        self._calculate_flattened_size(input_channels, input_size) # self.flatened_size
        self.fc_encoder = nn.Linear(self.flattened_size, latent_dim) # flattened_size -> code_size

        # CLASSIFICATION HEAD
        classifier_layers = []
        current_dim = latent_dim
        for hidden_dim in classification_hidden_dims:
            classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # DECODE
        self.fc_decoder = nn.Linear(latent_dim, self.flattened_size) # code_size -> flattened_size
        
        # decoder component
        # NOTE: using combination of Conv2d and Upsample (using bilinear interpolation) instead of ConvTranspose2d -> results in higher-quality reconstruction 
        print(f"Decoder Block")
        self.decoder = nn.ModuleList()
        current_channels = filters_per_layer[-1]
        for i in range(n_cnn_layers - 1):
            # out_channels = current_channels // 
            out_channels = filters_per_layer[-(i + 2)]
            self.decoder.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=1
                )
            )
            self.decoder.append(nn.BatchNorm2d(out_channels))
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.decoder.append(nn.Dropout(dropout))
            current_size = ((current_size - 1) * 1 + kernel_size - 2 * padding) * 2
            current_channels = out_channels
            print(f"Decoder: Layer {i+1} output size: {current_size}x{current_size}, channels: {current_channels}")

        # final block to reconstruct original image
        out_channels = current_channels // 2
        self.decoder.append(
            nn.Conv2d(
                in_channels=current_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,  
            )
        )
        self.decoder.append(nn.Upsample(size=(28, 28), mode="bilinear", align_corners=True))
        current_size = ((current_size - 1) * 1 + kernel_size - 2 * padding) * 2
        current_channels = input_channels
        print(f"Decoder: Layer {n_cnn_layers} output size: {28}x{28}, channels: {current_channels}")
        
        # Optimizer setup
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)

        # Loss function (MSE is typical for reconstruction)
        self.reconstruction_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()

    def _calculate_flattened_size(self, input_channels: int, input_size: int):
        x = torch.randn(1, input_channels, input_size, input_size)
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        x = x.view(x.size(0), -1)
        self.flattened_size = x.view(1, -1).size(1)
        # print(f"flattened size: {self.flattened_size}")

    def encode(self, x: torch.Tensor):
        # NxN -> nxn
        for i, layer in enumerate(self.encoder):
            x = layer(x)
        # print(f"encoder output: {x.shape}")
        x = x.view(x.size(0), -1) # flatten
        return self.fc_encoder(x)

    def decode(self, x: torch.Tensor):
        # nxn -> NxN
        x = self.fc_decoder(x)
        x =  x.view(x.size(0), self.filters_per_layer[-1], int((self.flattened_size // (self.filters_per_layer[-1])) ** 0.5), int((self.flattened_size // (self.filters_per_layer[-1])) ** 0.5))
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return torch.sigmoid(x) # to contrain output between [0,1]
    
    def classify(self, z: torch.Tensor):
        return self.classifier(z)

    def forward(self, x: torch.Tensor):
        # using encode and decode
        # print(f"input shape: {x.shape}")
        z = self.encode(x)
        # print(f"fc_encode output: {z.shape}")
        reconstructed = self.decode(z)
        classification = self.classify(z)
        # print(f"reconstructed output: {reconstructed.shape}")
        return reconstructed, classification
    

def validate_CNN_AutoEncoderClassifier(model: CNNAutoEncoder, val_dataloader: DataLoader, device: torch.device, alpha: float=0.5):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            reconstructed, classification = model(data)
            
            recon_loss = model.reconstruction_criterion(reconstructed, data)
            class_loss = model.classification_criterion(classification, target.float())
            loss = alpha * recon_loss + (1 - alpha) * class_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            pred = classification.argmax(dim=1)
            target_indices = target.argmax(dim=1)
            correct += pred.eq(target_indices).sum().item()
            total += target.size(0)

    return {
        "total_loss": total_loss / len(val_dataloader),
        "reconstruction_loss": total_recon_loss / len(val_dataloader),
        "classification_loss": total_class_loss / len(val_dataloader),
        "accuracy": 100. * correct / total
    }
    
def train_CNN_AutoEncoder(model: CNNAutoEncoder, num_epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, device: torch.device, alpha: float=0.5):
    print(f"Training CNN Autoencoder model...")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            model.optimizer.zero_grad()
            reconstructed, classification = model(data)

            recon_loss = model.reconstruction_criterion(reconstructed, data)
            class_loss = model.classification_criterion(classification, target.float())

            loss = alpha * recon_loss + (1 - alpha) * class_loss

            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            # Calculate accuracy
            pred = classification.argmax(dim=1)
            target_indices = target.argmax(dim=1)
            correct += pred.eq(target_indices).sum().item()
            total += target.size(0)
            
            if config["cnn_autoencoder"]["wandb_logging"]:
                wandb.log({
                    "train/epoch": epoch,
                    "train/total_loss": loss.item(),
                    "train/reconstruction_loss": recon_loss.item(),
                    "train/classification_loss": class_loss.item()
                })

            if (batch_idx + 1) % 100 == 0:
                print(f"batch: {batch_idx + 1}/{len(train_dataloader)}...")

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_class_loss = total_class_loss / len(train_dataloader)
        accuracy = 100. * correct / total

        # Validation
        val_metrics = validate_CNN_AutoEncoderClassifier(model, val_dataloader, device, alpha)
        
        if config["cnn_autoencoder"]["wandb_logging"]:
            wandb.log({
                "validation/total_loss": val_metrics["total_loss"],
                "validation/reconstruction_loss": val_metrics["reconstruction_loss"],
                "validation/classification_loss": val_metrics["classification_loss"],
                "validation/accuracy": val_metrics["accuracy"]
            })
        
        print(f"Train - Total Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, "
              f"Class Loss: {avg_class_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Val - Total Loss: {val_metrics['total_loss']:.4f}, Recon Loss: {val_metrics['reconstruction_loss']:.4f}, "
              f"Class Loss: {val_metrics['classification_loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
        print("-" * 90)


def evaluate_CNN_AutoEncoder(model: CNNAutoEncoder, test_dataloader: DataLoader, device: torch.device, alpha: float=0.5):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            reconstructed, classification = model(data)
            
            recon_loss = model.reconstruction_criterion(reconstructed, data)
            class_loss = model.classification_criterion(classification, target.float())
            loss = alpha * recon_loss + (1 - alpha) * class_loss
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            
            pred = classification.argmax(dim=1)
            target_indices = target.argmax(dim=1)
            correct += pred.eq(target_indices).sum().item()
            total += target.size(0)

    if config["cnn_autoencoder"]["wandb_logging"]:
        wandb.log({
            "test/total_loss": total_loss / len(test_dataloader),
            "test/reconstruction_loss": total_recon_loss / len(test_dataloader),
            "test/classification_loss": total_class_loss / len(test_dataloader),
            "test/accuracy": 100. * correct / total
        })

    return {
        "total_loss": total_loss / len(test_dataloader),
        "reconstruction_loss": total_recon_loss / len(test_dataloader),
        "classification_loss": total_class_loss / len(test_dataloader),
        "accuracy": 100. * correct / total
    }
