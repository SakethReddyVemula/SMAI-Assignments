import torch
import torch.nn as nn

class OCRModel(nn.Module):
    def __init__(
        self,
        image_height=64,
        image_width=256,
        batch_size=32,
        max_word_length=20,
        hidden_size=256,
        num_layers=2,
        lr=1e-3,
        vocab_size=10000
    ):
        super(OCRModel, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.max_word_length = max_word_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.vocab_size = vocab_size
        


        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        
        # Calculate the size of flattened features
        self.feature_size = 256 * (self.image_height // 8) * (self.image_width // 8)
        
        # Linear layer to reduce features
        self.feature_reducer = nn.Linear(self.feature_size, self.hidden_size)
        
        # RNN Decoder
        self.rnn = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        batch_size = x.size(0)
        features = self.cnn(x)
        features = features.view(batch_size, -1)
        features = self.feature_reducer(features)
        
        # Expand features for RNN
        features = features.unsqueeze(1).repeat(1, self.max_word_length, 1)
        
        # RNN decoding
        rnn_out, _ = self.rnn(features)
        
        # Generate character probabilities
        output = self.fc(rnn_out)
        return output