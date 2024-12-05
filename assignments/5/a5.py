import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath('../../models/KDE'))
sys.path.append(os.path.abspath('../../models/gmm'))

from kde import KDE
from gmm import GMM

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../../models/HMM"))
from hmm import DigitHMMRecognizer

# RNN
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../../models/RNN'))
from rnn import BitCounterRNN

# OCR
import string
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict
import torch.optim as optim
from nltk.corpus import words
import torchvision.transforms as transforms
sys.path.append(os.path.abspath('../../models/RNN'))
from ocr import OCRModel
from datetime import datetime

class WordImageDataset(Dataset):
    def __init__(self, word_list: List[str], transform=None):
        self.words = word_list
        self.transform = transform
        self.font = ImageFont.truetype("DejaVuSans.ttf", 32)
    
    def __len__(self):
        return len(self.words)
    
    def render_word(self, word: str) -> Image.Image:
        """Render a word as a PNG image."""
        img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
        draw = ImageDraw.Draw(img)
        
        # Center the text
        bbox = draw.textbbox((0, 0), word, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (IMAGE_WIDTH - text_width) // 2
        y = (IMAGE_HEIGHT - text_height) // 2
        
        draw.text((x, y), word, fill=0, font=self.font)
        return img
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        word = self.words[idx].lower()
        
        # Create image
        img = self.render_word(word)
        if self.transform:
            img = self.transform(img)
        
        # Create target sequence
        target = torch.zeros(MAX_WORD_LENGTH, dtype=torch.long)
        for i, char in enumerate(word):
            if i < MAX_WORD_LENGTH:
                target[i] = CHAR_TO_IDX.get(char, 0)
        
        return img, target, len(word)

class Metrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct_chars = 0
        self.total_chars = 0
        self.word_matches = 0
        self.total_words = 0
        self.predictions = []  # Store (ground_truth, prediction) pairs
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               lengths: torch.Tensor) -> None:
        """Update metrics with batch results."""
        batch_size = predictions.size(0)
        
        for i in range(batch_size):
            pred = predictions[i, :lengths[i]]
            target = targets[i, :lengths[i]]
            
            # Character-level accuracy
            self.correct_chars += (pred == target).sum().item()
            self.total_chars += lengths[i].item()
            
            # Word-level accuracy
            if torch.all(pred == target):
                self.word_matches += 1
            self.total_words += 1
            
            # Store prediction examples
            if len(self.predictions) < 10:  # Store first 10 examples
                pred_word = ''.join([IDX_TO_CHAR[idx.item()] for idx in pred])
                target_word = ''.join([IDX_TO_CHAR[idx.item()] for idx in target])
                self.predictions.append((target_word, pred_word))
    
    def get_metrics(self) -> Dict[str, float]:
        return {
            'char_accuracy': self.correct_chars / max(1, self.total_chars),
            'word_accuracy': self.word_matches / max(1, self.total_words)
        }
    

def generate_random_baseline(targets: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch_size = targets.size(0)
    random_preds = torch.zeros_like(targets)
    
    for i in range(batch_size):
        length = lengths[i].item()
        random_preds[i, :length] = torch.randint(0, VOCAB_SIZE, (length,))
    
    return random_preds

def validate_model(model: nn.Module, val_loader: DataLoader, 
                  device: torch.device, epoch: int) -> Dict[str, float]:
    model.eval()
    metrics = Metrics()
    random_metrics = Metrics()
    
    print(f"\nValidation Report - Epoch {epoch}")
    print("-" * 50)
    
    with torch.no_grad():
        for batch_idx, (images, targets, lengths) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Model predictions
            outputs = model(images)
            predictions = outputs.argmax(dim=2)
            metrics.update(predictions, targets, lengths)
            
            # Random baseline predictions
            random_preds = generate_random_baseline(targets, lengths)
            random_metrics.update(random_preds, targets, lengths)
    
    # Calculate metrics
    model_metrics = metrics.get_metrics()
    baseline_metrics = random_metrics.get_metrics()
    
    # Print detailed report
    print("\nModel Performance:")
    print(f"Character Accuracy: {model_metrics['char_accuracy']:.4f}")
    print(f"Word Accuracy: {model_metrics['word_accuracy']:.4f}")
    
    print("\nRandom Baseline:")
    print(f"Character Accuracy: {baseline_metrics['char_accuracy']:.4f}")
    print(f"Word Accuracy: {baseline_metrics['word_accuracy']:.4f}")
    
    print("\nPrediction Examples:")
    print("-" * 50)
    print("Ground Truth | Prediction")
    print("-" * 50)
    for target, pred in metrics.predictions:
        print(f"{target:<12} | {pred}")
    
    return model_metrics

def save_training_report(epoch: int, train_metrics: Metrics, 
                        val_metrics: Metrics, history: List[Dict[str, float]]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'training_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write("OCR Model Training Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Training completed after {epoch + 1} epochs\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("Final Metrics:\n")
        f.write(f"Training Character Accuracy: {train_metrics.get_metrics()['char_accuracy']:.4f}\n")
        f.write(f"Validation Character Accuracy: {val_metrics['char_accuracy']:.4f}\n")
        f.write(f"Training Word Accuracy: {train_metrics.get_metrics()['word_accuracy']:.4f}\n")
        f.write(f"Validation Word Accuracy: {val_metrics['word_accuracy']:.4f}\n\n")
        
        f.write("Example Predictions:\n")
        f.write("-" * 50 + "\n")
        f.write("Ground Truth | Prediction\n")
        f.write("-" * 50 + "\n")
        for target, pred in train_metrics.predictions:
            f.write(f"{target:<12} | {pred}\n")
        
        f.write("\nTraining History:\n")
        f.write("-" * 50 + "\n")
        for epoch, metrics in enumerate(history):
            f.write(f"Epoch {epoch + 1}:\n")
            f.write(f"Character Accuracy: {metrics['char_accuracy']:.4f}\n")
            f.write(f"Word Accuracy: {metrics['word_accuracy']:.4f}\n")
            f.write("-" * 25 + "\n")


def train_ocr_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader, 
        num_epochs: int,
        device: torch.device
    ) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    patience=2, factor=0.5)
    
    model.to(device)
    best_char_acc = 0
    metrics_history = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metrics = Metrics()
        
        for batch_idx, (images, targets, lengths) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Reshape for loss calculation
            batch_size = outputs.size(0)
            outputs_flat = outputs.view(-1, VOCAB_SIZE)
            targets_flat = targets.view(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            # Update training metrics
            predictions = outputs.argmax(dim=2)
            train_metrics.update(predictions, targets, lengths)
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                metrics = train_metrics.get_metrics()
                print(f'Epoch [{epoch+1}/{num_epochs}] '
                      f'Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Char Acc: {metrics["char_accuracy"]:.4f}')
        
        # Validation phase
        val_metrics = validate_model(model, val_loader, device, epoch)
        metrics_history.append(val_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['char_accuracy'])
        
        # Save best model
        if val_metrics['char_accuracy'] > best_char_acc:
            best_char_acc = val_metrics['char_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, 'best_ocr_model.pth')
        
        # Save training report
        save_training_report(epoch, train_metrics, val_metrics, metrics_history)



class BinarySequenceDataset(Dataset):
    def __init__(self, num_sequences, min_length=1, max_length=16):
        self.sequences = []
        self.counts = []

        for _ in range(num_sequences):
            length = np.random.randint(min_length, max_length + 1)
            sequence = np.random.randint(0, 2, size=length)
            count = np.sum(sequence)

            self.sequences.append(torch.FloatTensor(sequence))
            self.counts.append(torch.FloatTensor([count]))

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.counts[idx]
    
def collate_fn(batch):
    sequences, counts = zip(*batch)

    sequences_packed = nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
    counts = torch.stack(counts)

    return sequences_packed, counts

def evaluate_generalization(model, min_length=1, max_length=32, samples_per_length=1000, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    mae_values = []
    
    for length in range(min_length, max_length + 1):
        dataset = BinarySequenceDataset(samples_per_length, length, length)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
        
        total_mae = 0
        with torch.no_grad():
            for sequences, counts in dataloader:
                if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):
                    sequences = sequences.to(device)
                else:
                    sequences = sequences.to(device)
                counts = counts.to(device)
                outputs = model(sequences)
                mae = torch.abs(outputs - counts).mean().item()
                total_mae += mae
                
        mae_values.append(total_mae / len(dataloader))
    
    return mae_values

# Random baseline for comparison
def random_baseline(min_length=1, max_length=32, samples_per_length=1000):
    mae_values = []
    
    for length in range(min_length, max_length + 1):
        true_counts = np.random.binomial(length, 0.5, samples_per_length)
        pred_counts = np.random.randint(0, length + 1, samples_per_length)
        mae = np.mean(np.abs(true_counts - pred_counts))
        mae_values.append(mae)
    
    return mae_values

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None):
    # Automatically detect the available device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    baseline_mae = random_baseline(min_length=1, max_length=16, samples_per_length=625)
    print(f"Random Baseline MAE: {baseline_mae}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for sequences, counts in train_loader:
            # Move packed sequence to device
            if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):
                sequences = sequences.to(device)
            else:
                sequences = sequences.to(device)
            counts = counts.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, counts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        model.eval()
        val_loss = 0
        total_mae = 0
        with torch.no_grad():
            for sequences, counts in val_loader:
                if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):
                    sequences = sequences.to(device)
                else:
                    sequences = sequences.to(device)
                counts = counts.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, counts).item()
                mae = torch.absolute(outputs - counts).mean().item()  
                total_mae += mae    
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val MAE: {(total_mae / len(val_loader)):.4f}')
    
    return train_losses, val_losses

def plot_generalization_results(mae_values, baseline_mae=None):
    plt.figure(figsize=(10, 6))
    
    # Plot model MAE
    x = range(1, len(mae_values) + 1)
    plt.plot(x, mae_values, marker='o', label='Model MAE', linewidth=2)
    
    # Plot baseline if provided
    if baseline_mae is not None:
        plt.plot(x, baseline_mae, marker='x', linestyle='--', label='Random Baseline', linewidth=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Performance vs Sequence Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add x-axis ticks for each sequence length
    plt.xticks(x)
    
    plt.tight_layout()
    # plt.savefig("figures/Counting_Bits_seqlen_vs_MAE.png")
    plt.show()




# Generate synthetic data
def generate_synthetic_data():
    # Generate larger diffused circle (3000 points)
    n_points_large = 3000
    radius_large = 1.5
    theta_large = 2 * np.pi * np.random.random(n_points_large)
    r_large = radius_large * np.sqrt(np.random.random(n_points_large))
    x_large = r_large * np.cos(theta_large)
    y_large = r_large * np.sin(theta_large)
    
    # Add noise to simulate variation
    noise_large = np.random.normal(0, 0.2, (n_points_large, 2))
    points_large = np.column_stack((x_large, y_large)) + noise_large
    
    # Generate smaller dense circle (500 points)
    n_points_small = 500
    radius_small = 0.3
    theta_small = 2 * np.pi * np.random.random(n_points_small)
    r_small = radius_small * np.sqrt(np.random.random(n_points_small))
    x_small = r_small * np.cos(theta_small) + 1.0
    y_small = r_small * np.sin(theta_small) + 0.5
    
    # Add less noise to the dense circle
    noise_small = np.random.normal(0, 0.05, (n_points_small, 2))
    points_small = np.column_stack((x_small, y_small)) + noise_small
    
    # Combine the points
    points = np.vstack((points_large, points_small))
    
    return points


def plot_original_data(data):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=1)
    plt.title('Original Data')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_gmm_fit(data, gmm, n_components):
    plt.figure(figsize=(10, 8))
    
    # Fit GMM and get predictions
    gmm.fit(data)
    predictions = gmm.predict(data)
    
    # Plot data points colored by cluster
    scatter = plt.scatter(data[:, 0], data[:, 1], c=predictions, 
                         cmap='viridis', alpha=0.5, s=1)
    
    # Plot cluster centers
    means = gmm.means
    plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100, 
                linewidths=3, label='Cluster Centers')
    
    plt.title(f'GMM with {n_components} Components')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster Assignment')
    plt.savefig(f"figures/gmm_n_comp_{n_components}.png")
    plt.show()
    
    # Print performance metrics
    print(f"\nGMM Performance Metrics for {n_components} components:")
    print(f"Silhouette Score: {gmm.calculate_silhouette_score(data):.3f}")
    print(f"Davies-Bouldin Score: {gmm.calculate_davies_bouldin_score(data):.3f}")
    print(f"Final Log-Likelihood: {gmm.log_likelihood:.3f}")

# HMM
def load_digit_files(digit, data_dir="../../data/interim/5/archive/training-recordings"):
    pattern = os.path.join(data_dir, f"{digit}_*.wav")
    files = sorted(glob(pattern))
    if not files:
        raise ValueError(f"No files found for digit {digit} in {data_dir}")
    return files

def load_test_files(data_dir="../../data/interim/5/archive/testing-recordings"):
    pattern = os.path.join(data_dir, "*.wav")
    files = sorted(glob(pattern))
    if not files:
        raise ValueError(f"No test files found in {data_dir}")
    return files

def get_test_labels(test_files):
    labels = []
    for file in test_files:
        # Extract digit from filename (e.g., "7_jackson_32.wav" -> "7")
        digit = os.path.basename(file).split('_')[0]
        labels.append(digit)
    return labels

def evaluate_hmm_model(recognizer, test_files, true_labels, print_predictions: bool=False):
    """Evaluate model performance"""
    predictions = []
    scores = []
    
    for file in test_files:
        try:
            features = recognizer.extract_features(file)
            pred = recognizer.predict(features)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            predictions.append(None)
    
    # Filter out None predictions
    valid_predictions = [(p, t) for p, t in zip(predictions, true_labels) if p is not None]
    if not valid_predictions:
        return 0.0
    
    if print_predictions:
        for pred_label, true_label in valid_predictions:
            print(f"Predicted Label: {pred_label}\tActual Label: {true_label}")

    pred_labels, true_labels = zip(*valid_predictions)
    
    accuracy = np.mean(np.array(pred_labels) == np.array(true_labels))
    return accuracy


if __name__ == "__main__":
    TASK_2_2 = False
    TASK_2_3 = False
    TASK_3 = False
    TASK_4 = False
    TASK_5 = True
    if TASK_2_2:
        # Generate synthetic data
        data = generate_synthetic_data()

        # Create and fit KDE
        kde = KDE(kernel='gaussian', bandwidth=0.5)
        kde.fit(data)

        # Visualize the results
        kde.visualize(data)

    elif TASK_2_3:
        # Generate synthetic data
        data = generate_synthetic_data()

        # Plot original data
        print("Plotting original data...")
        plot_original_data(data)

        # Fit and plot KDE
        print("\nFitting and plotting KDE...")
        kde = KDE(kernel='gaussian', bandwidth=0.2)
        kde.fit(data)
        kde.visualize(data)

        # Fit and plot GMMs with different components
        n_components_list = [2, 3, 4]
        for n_comp in n_components_list:
            print(f"\nFitting and plotting GMM with {n_comp} components...")
            gmm = GMM(num_components=n_comp, max_iterations=100, threshold=1e-6)
            plot_gmm_fit(data, gmm, n_comp)

    elif TASK_3:
        # Initialize recognizer
        recognizer = DigitHMMRecognizer(n_states=5)

        # For each digit (0-9):
        for digit in range(10):
            # Load training files for this digit
            files = load_digit_files(digit)
            features_list = [recognizer.extract_features(f) for f in files]
            recognizer.train_model(str(digit), features_list)

        # Evaluate
        test_files = load_test_files()
        true_labels = get_test_labels(test_files)
        accuracy = evaluate_hmm_model(recognizer, test_files, true_labels, print_predictions=True)
        print(f"Test accuracy: {accuracy:.2%}")

        # Visualize features for a sample
        # sample_features = recognizer.extract_features(test_files[0])
        # recognizer.visualize_features(sample_features, "Sample Digit MFCC Features")

        # for digit in range(10):
        #     for speaker_id in range(6):
        #         sample_features = recognizer.extract_features(test_files[digit * 30 + (5 * speaker_id)])
        #         recognizer.visualize_features(sample_features, digit, speaker_id)

        # Evaluate on self recordings
        self_test_files = load_test_files("../../data/interim/5/archive/self_recordings")
        self_true_labels = get_test_labels(self_test_files)
        accuracy = evaluate_hmm_model(recognizer, self_test_files, self_true_labels, print_predictions=True)
        print(f"Test Accuracy on Self Recordings: {accuracy:.2%}")
        
        for digit, test_file in enumerate(self_test_files):
            sample_features = recognizer.extract_features(test_file)
            recognizer.visualize_features(sample_features, digit, 6)

    elif TASK_4:
        train_dataset = BinarySequenceDataset(80000)
        val_dataset = BinarySequenceDataset(10000)
        test_dataset = BinarySequenceDataset(10000)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

        # Initialize model and training components
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BitCounterRNN(
            input_size=1,
            hidden_size=16,
            num_layers=2,
            dropout=0.2
        )
        criterion = nn.L1Loss()  # MAE loss
        optimizer = torch.optim.Adam(model.parameters())

        # Train the model
        train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer)

        # Evaluate generalization
        mae_values = evaluate_generalization(model)
        baseline_mae = random_baseline()
        print(f"mae_value:\n{mae_values}")
        print(f"baseline_mae:\n{baseline_mae}")

        # Create the plot
        plot_generalization_results(mae_values, baseline_mae)
    elif TASK_5:
        # Constants
        IMAGE_HEIGHT = 64
        IMAGE_WIDTH = 256
        BATCH_SIZE = 32
        MAX_WORD_LENGTH = 20
        HIDDEN_SIZE = 256
        NUM_LAYERS = 2
        LEARNING_RATE = 0.001

        # Character vocabulary
        VOCAB = string.ascii_lowercase + ' '
        CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
        IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}
        VOCAB_SIZE = len(VOCAB)

        # Get word list from NLTK
        word_list = [w.lower() for w in words.words() if len(w) <= MAX_WORD_LENGTH]
        word_list = word_list[:10000]  # Take first 100k words
        
        # Split into train and validation sets
        train_words = word_list[:9000]
        val_words = word_list[9000:]
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Create datasets
        train_dataset = WordImageDataset(train_words, transform=transform)
        val_dataset = WordImageDataset(val_words, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Initialize model
        model = OCRModel(
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            batch_size=BATCH_SIZE,
            max_word_length=MAX_WORD_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            lr=LEARNING_RATE,
            vocab_size=VOCAB_SIZE
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Train model
        train_ocr_model(model, train_loader, val_loader, num_epochs=5, device=device)



