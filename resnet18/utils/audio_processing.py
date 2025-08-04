import librosa
import numpy as np
import torch
import cv2
from pathlib import Path
import soundfile as sf
import gc
from utils.config import Config

class AudioProcessor:
    def __init__(self):
        self.config = Config()
        
    def load_audio(self, file_path, max_length=None):
        """Load and preprocess audio file with memory optimization"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Pad or truncate to fixed length
            if max_length:
                target_length = max_length * self.config.SAMPLE_RATE
                if len(audio) > target_length:
                    audio = audio[:target_length]
                else:
                    audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
            # Ensure audio is float32 to save memory
            audio = audio.astype(np.float32)
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def audio_to_melspectrogram(self, audio, img_size=224):
        """Convert audio to mel-spectrogram image with memory optimization"""
        try:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.SAMPLE_RATE,
                n_mels=self.config.N_MELS,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-255 with uint8 to save memory
            mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                            (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
            
            # Clear intermediate variables
            del mel_spec, mel_spec_db
            gc.collect()
            
            # Resize to target image size
            mel_spec_resized = cv2.resize(mel_spec_norm, (img_size, img_size))
            
            # Convert to 3-channel image (RGB)
            mel_spec_rgb = cv2.applyColorMap(mel_spec_resized, cv2.COLORMAP_VIRIDIS)
            
            # Clear intermediate variables
            del mel_spec_norm, mel_spec_resized
            gc.collect()
            
            return mel_spec_rgb
            
        except Exception as e:
            print(f"Error converting audio to mel-spectrogram: {e}")
            return None
    
    def audio_to_mfcc(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mfcc=n_mfcc,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH
        )
        return mfcc
    
    def audio_to_spectral_features(self, audio):
        """Extract various spectral features"""
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.config.SAMPLE_RATE
        ).flatten()
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.config.SAMPLE_RATE
        ).flatten()
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(audio).flatten()
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=self.config.SAMPLE_RATE
        ).flatten()
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=audio, sr=self.config.SAMPLE_RATE
        ).flatten()
        
        return features
    
    def get_dataset_paths(self):
        """Get all audio file paths organized by split and label"""
        dataset_structure = {
            'training': {'fake': [], 'real': []},
            'validation': {'fake': [], 'real': []},
            'testing': {'fake': [], 'real': []}
        }
        
        for split in ['training', 'validation', 'testing']:
            for label in ['fake', 'real']:
                split_path = self.config.DATASET_PATH / split / label
                if split_path.exists():
                    audio_files = list(split_path.glob('*.wav')) + list(split_path.glob('*.mp3'))
                    dataset_structure[split][label] = audio_files
                    print(f"Found {len(audio_files)} {label} files in {split}")
                else:
                    print(f"Warning: Path {split_path} does not exist")
        
        return dataset_structure