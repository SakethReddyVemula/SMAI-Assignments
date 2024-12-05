import numpy as np
from librosa import feature, load
import soundfile as sf
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DigitHMMRecognizer:
    def __init__(self, n_states=5):
        self.models = {}
        self.n_states = n_states
        
    def extract_features(self, audio_file):
        """Extract MFCC features from audio file with appropriate parameters"""
        # Load audio file with specified sample rate
        signal, sr = load(audio_file, sr=8000)  # Force 8kHz sample rate
        
        # Set parameters appropriate for short speech segments
        n_fft = 256  # Smaller window size
        hop_length = n_fft // 4  # 75% overlap
        
        # Extract MFCC features with delta and delta-delta
        mfccs = feature.mfcc(
            y=signal, 
            sr=sr, 
            n_mfcc=13,  # Number of MFCC coefficients
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Compute delta features
        delta_mfccs = feature.delta(mfccs)
        delta2_mfccs = feature.delta(mfccs, order=2)
        
        # Combine all features
        combined_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        # Transpose to get time steps as rows and features as columns
        features = combined_features.T
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-12)
        
        return features
    
    def train_model(self, digit, features_list):
        """Train HMM model for a specific digit"""
        # Initialize HMM with init_params excluding 's' and 't'
        model = hmm.GaussianHMM(
            n_components=self.n_states, 
            covariance_type="diag",
            n_iter=100,
            random_state=42,  # For reproducibility
            init_params='mc'  # Only initialize means and covariances, not states or transitions
        )
        
        # Concatenate all features for training
        X = np.vstack(features_list)
        lengths = [x.shape[0] for x in features_list]
        
        # Initialize parameters to help convergence
        model.startprob_ = self._generate_start_prob(self.n_states)
        model.transmat_ = self._generate_left_right_transmat(self.n_states)
        
        # Fit the model
        model.fit(X, lengths=lengths)
        self.models[digit] = model
        
    def _generate_start_prob(self, n_states):
        """Generate initial state probabilities favoring earlier states"""
        startprob = np.zeros(n_states)
        startprob[0] = 0.8  # High probability of starting in first state
        startprob[1] = 0.2  # Small probability of starting in second state
        return startprob
    
    def _generate_left_right_transmat(self, n_states, stay_prob=0.6):
        """Generate left-right transition matrix"""
        transmat = np.zeros((n_states, n_states))
        for i in range(n_states):
            if i == n_states - 1:
                transmat[i, i] = 1.0  # Stay in final state
            else:
                transmat[i, i] = stay_prob  # Probability of staying in current state
                transmat[i, i + 1] = 1 - stay_prob  # Probability of moving to next state
        return transmat
    
    def predict(self, features):
        """Predict digit from features"""
        scores = {}
        for digit, model in self.models.items():
            try:
                score = model.score(features)
                scores[digit] = score
            except Exception as e:
                print(f"Warning: Error scoring digit {digit}: {str(e)}")
                scores[digit] = float('-inf')
        
        # Add score normalization
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        probs = {k: v/total for k, v in exp_scores.items()}
        
        return max(probs.items(), key=lambda x: x[1])[0]
    
    def visualize_features(self, features, digit=0, speaker_id=0):
        """Visualize MFCC features as a heatmap"""
        plt.figure(figsize=(10, 4))
        plt.imshow(features.T, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"MFCC Spectrogram digit:{digit} speaker:{speaker_id}")
        plt.xlabel('Time Frames')
        plt.ylabel('Features')
        # plt.savefig(f"../../assignments/5/figures/MFCC_sp_digit_{digit}_speaker_{speaker_id}.png")
        plt.show()
