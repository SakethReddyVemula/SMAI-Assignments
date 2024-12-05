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

def parse_folder_name(folder_name: str):
    list_format = [int(d) for d in str(folder_name)]
    if config["task"] == "single_label":
        if folder_name == "0":
            return []
        else:
            return [(len(list_format))]
    elif config["task"] == "multi_label":
        if folder_name == "0":
            return []
        else:
            return list_format


def load_mnist_data(path_to_data: str):
    data = []

    for folder_name in os.listdir(path_to_data):
        folder_path = os.path.join(path_to_data, folder_name)
        if not os.path.isdir(folder_path):
            continue

        digits = parse_folder_name(folder_name)

        if config["task"] == "single_label":
            label = get_one_hot_encoding(digits, num_classes=config["slc"]["num_classes"])
        elif config["task"] == "multi_label":
            label = get_one_hot_encoding(digits, num_classes=config["mlc"]["num_classes"])

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            data.append((img_path, label))

    return data


class MultiMNISTDataset(Dataset):
    def __init__(self, split: str="train", transform=None):
        super(MultiMNISTDataset, self).__init__()

        self.transform = transform
        if split == "train":
            path_to_data = str(config["train_folder"])
        elif split == "val":
            path_to_data = str(config["val_folder"])
        elif split == "test":
            path_to_data = str(config["test_folder"])
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.data = load_mnist_data(path_to_data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L') # greyscale
        if self.transform:
            image = self.transform(image)
        else: # default
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.unsqueeze(0) 
        
        return image, label
    
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
    
train_dataset = MultiMNISTDataset(split="train", transform=transform)
val_dataset = MultiMNISTDataset(split="val", transform=transform)
test_dataset = MultiMNISTDataset(split="test", transform=transform)

if config["task"] == "single_label":
    train_dataloader = DataLoader(train_dataset, batch_size=config["slc"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["slc"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["slc"]["batch_size"], shuffle=True)
elif config["task"] == "multi_label":
    train_dataloader = DataLoader(train_dataset, batch_size=config["mlc"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["mlc"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["mlc"]["batch_size"], shuffle=True)


class CNN(nn.Module):
    def __init__(
            self,
            task: str,
            label_type: str,
            lr: float=1e-4,
            dropout: float=0.1,
            n_cnn_layers: int=3,
            optimizer: str="AdamW",
            save_feature_maps: bool=False,
            input_channels: int=1,
            num_classes: int=3,
            kernel_size: int=3,
            padding: int=1
    ):
        super(CNN, self).__init__()
        self.lr = lr
        self.task = task
        self.label_type = label_type
        self.save_feature_maps = save_feature_maps
        # self.feature_maps = defaultdict(list) if save_feature_maps else None
        self.features_maps: list = []

        current_size = config["input_size"]  # Initial image size
        current_channels = input_channels
        
        self.layers = nn.ModuleList()
        
        for i in range(n_cnn_layers):
            out_channels = current_channels * 2 if i > 0 else 32
            self.layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels, 
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            current_size = (current_size + 2 * padding - kernel_size) // 1 + 1
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU())            
            if current_size // 2 >= 1:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size //= 2
            self.layers.append(nn.Dropout(dropout))
            current_channels = out_channels
            print(f"Layer {i+1} output size: {current_size}x{current_size}, channels: {current_channels}")

        self.feature_size = current_channels * current_size * current_size
        print(f"Feature size before FC layer: {self.feature_size}")

        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.task == "classification":
            if self.label_type == "multi":
                self.output_layer = nn.Sequential(
                    nn.Linear(512, num_classes),
                    nn.Sigmoid()
                )
            else:  # single
                self.output_layer = nn.Sequential(
                    nn.Linear(512, num_classes),
                    nn.Softmax(dim=1)
                )
        else:  # regression
            self.output_layer = nn.Sequential(
                nn.Linear(512, num_classes),
                nn.Identity()
            )

        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer.lower() == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)

        if self.task == "classification":
            if self.label_type == "single":
                self.criterion = nn.CrossEntropyLoss()
            else:  # multi
                self.criterion = nn.BCELoss()
        else:  # regression
            self.criterion = nn.MSELoss()

    def _save_feature_map(self, x: torch.Tensor, layer_name: str):
        if self.save_feature_maps:
            self.feature_maps[layer_name].append(x.detach().cpu())

    def forward(self, x: torch.Tensor):
        if self.save_feature_maps:
            self.features_maps = []
            
        # Conv layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i % 4 == 0 and i != 0 and self.save_feature_maps:
                self.features_maps.append(x.detach().cpu())

        # # Flatten
        x = x.view(-1, self.feature_size)
        
        # # FC layers
        x = self.fc_layers(x)
        
        # # Output layer
        x = self.output_layer(x)
        
        return x
    
def validate_CNN(model: CNN, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    total_val_loss = 0
    
    if config["task"] == "multi_label" and model.task == "classification":
        total_em_count = 0 # exact match
        total_em_samples = 0
        total_m_count = 0 # match (not exact)
        total_m_samples = 0
        total_hamming_count = 0
    else:
        metric = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if model.task == "classification":
                loss = model.criterion(output, target.float())
                if config["task"] == "single_label":
                    pred = output.argmax(dim=1)
                    target = target.argmax(dim=1)
                    metric += (pred == target).sum().item()
                elif config["task"] == "multi_label":
                    pred = (output > 0.5).float()
                    total_m_count += (pred == target).sum().item()
                    total_em_count += (pred == target).all(dim=1).sum().item() 
                    total_m_samples += data.size(0) * target.size(1) # 32 * 10
                    total_em_samples += data.size(0) # 32
                    total_hamming_count += ((pred != target).float().mean(dim=1).sum().item())
                    # print(f"pred: {pred}")
                    # print(f"total_m_count: {total_m_count}")
                    # print(f"total_em_count: {total_em_count}")
                    # print(f"total_m_samples: {total_m_samples}")
                    # print(f"total_em_samples: {total_em_samples}")
                    # print(f"total_hamming_count: {total_hamming_count}")
            else:
                loss = model.criterion(output.squeeze(), target.float())
                mse = ((output - target) ** 2).mean().item()
                # metric += loss.item()
                metric += mse * target.size(0)

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        if model.task == "classification":
            if config["task"] == "single_label":
                # print(f"hello from cs")
                metric = 100.00 * (metric / len(val_dataloader.dataset)) # accuracy
            elif config["task"] == "multi_label":
                # print(f"hello from cm")
                match_accuracy = 100.00 * (total_m_count / total_m_samples) if total_m_count > 0 else 0.00
                exact_match_accuracy = 100.00 * (total_em_count / total_em_samples) if total_em_count > 0 else 0.00
                hamming_loss = (total_hamming_count / total_em_samples if total_em_samples > 0 else 0.0)
        else:
            # print(f"hello from r")
            metric = metric / len(val_dataloader) # MSE
        
        if config["task"] == "multi_label" and model.task == "classification":
            return avg_val_loss, match_accuracy, exact_match_accuracy, hamming_loss
        else:
            return avg_val_loss, metric


def train_CNN(model: CNN, num_epochs: int, train_dataloader: DataLoader, val_dataloader: DataLoader, device: torch.device):
    print(f"Training CNN {model.task} model...")
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            # print(f"data: {data}")
            # print(f"target: {target.shape}")
            data, target = data.to(device), target.to(device)
            model.optimizer.zero_grad()
            output = model(data)

            if model.task == "classification":
                loss = model.criterion(output, target.float())
            else:
                # print(f"output.shape: {output.shape}")
                # print(f"output.squeeze.shape: {output.squeeze().shape}")
                # print(f"target.shape: {target.shape}")
                loss = model.criterion(output.squeeze(), target.float())

            loss.backward()
            model.optimizer.step()

            total_loss += loss.item()

            if config["wandb_logging"]:
                wandb.log({
                    "train/epoch": epoch,
                    "train/loss": loss.item()
                })

            if (batch_idx + 1) % 100 == 0:
                print(f"batch: {batch_idx + 1}/{len(train_dataloader)}...")


        avg_epoch_loss = total_loss / len(train_dataloader)

        if config["task"] == "multi_label" and model.task == "classification":
            val_loss, match_accuracy, exact_match_accuracy, hamming_loss = validate_CNN(model, val_dataloader=val_dataloader, device=device)
        else:
            val_loss, val_metric = validate_CNN(model, val_dataloader, device)

        if config["wandb_logging"]:
            if model.task == "classification":
                if config["task"] == "single_label":
                    wandb.log({
                        "validation/loss": val_loss,
                        "validation/accuracy": val_metric
                    })
                elif config["task"] == "multi_label":
                    wandb.log({
                        "validation/loss": val_loss,
                        "validation/accuracy":  match_accuracy,
                        "validation/exact_match_accuracy": exact_match_accuracy,
                        "validation/hamming_loss": hamming_loss
                    })
            else:
                wandb.log({
                    "validation/loss": val_loss,
                    "validation/MSE": val_metric
                })


        print(f"Train loss: {avg_epoch_loss:.4f}")
        print(f"Validation loss: {val_loss}")
        
        if config["task"] == "multi_label" and model.task == "classification":
            print(f"Validation Accuracy: {match_accuracy}\tExact Match Accuracy: {exact_match_accuracy}\tHamming Loss: {hamming_loss}")
        else:
            print(f'Validation {"Accuracy" if model.task == "classification" else "MSE"}: {val_metric:.4f}')
        print(f"-"*90)


def evaluate_CNN(model: CNN, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    if config["task"] == "multi_label" and model.task == "classification":
        results = {
            'loss': 0.0,
            'match_accuracy': 0.0,
            'exact_match_accuracy': 0.0,
            'hamming_loss': 0.0
        }

        total_em_samples = 0
        total_m_samples = 0
    else:
        results = {
            'loss': 0.0,
            'metric': 0.0
        }

    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            if model.task == "classification":
                loss = model.criterion(output, target.float())
                if config['task'] == "multi_label":
                    pred = (output > 0.5).float()
                    results['match_accuracy'] += (pred == target).sum().item()
                    results['exact_match_accuracy'] += (pred == target).all(dim=1).sum().item()
                    results['hamming_loss'] += ((pred != target).float().mean(dim=1).sum().item())
                    total_m_samples += data.size(0) * target.size(1) # 32x10
                    total_em_samples += data.size(0)
                elif config["task"] == "single_label":
                    pred = output.argmax(dim=1)
                    target = target.argmax(dim=1)
                    results['metric'] += (pred == target).sum().item()
            else:
                loss = model.criterion(output.squeeze(), target.float())
                mse = ((output - target) ** 2).mean().item()
                results['metric'] += (mse * target.size(0))
        
            results['loss'] += loss.item()

    results['loss'] = results['loss'] / len(test_dataloader)
    
    if model.task == "classification":
        if config["task"] == "single_label":
            results['metric'] = 100.00 * (results['metric'] / len(test_dataloader.dataset))
        elif config["task"] == "multi_label":
            results['match_accuracy'] = 100.00 * (results['match_accuracy'] / total_m_samples) if total_m_samples > 0 else 0.00
            results['exact_match_accuracy'] = 100.00 * (results['exact_match_accuracy'] / total_em_samples) if total_em_samples > 0 else 0.0
            results['hamming_loss'] = (results['hamming_loss'] / total_em_samples) if total_em_samples > 0 else 0.0
    else:
        results['metric'] = results['metric'] / len(test_dataloader)

    if config["wandb_logging"]:
        if model.task == "classification":
            if config["task"] == "single_label":
                wandb.log({
                    "test/loss": results["loss"],
                    "test/accuracy": results['metric']
                })
            elif config["task"] == "multi_label":
                wandb.log({
                    "test/loss": results["loss"],
                    "test/accuracy": results['match_accuracy'],
                    "test/exact_match_accuracy": results['exact_match_accuracy'],
                    "test/hamming_loss": results['hamming_loss']
                })
        else:
            wandb.log({
                "test/loss": results['loss'],
                "test/MSE": results['metric']
            })

    return results