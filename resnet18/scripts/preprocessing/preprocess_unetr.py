import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.audio_processing import AudioProcessor

class UNETRPreprocessor:
    def __init__(self, batch_size=8):
        self.config = Config()
        self.audio_processor = AudioProcessor()
        self.config.create_directories()
        self.batch_size = batch_size  # Process files in batches to reduce memory usage
        
    def create_patches(self, mel_spec, patch_size=16):
        """Create patches for UNETR input"""
        h, w = mel_spec.shape
        
        # Ensure dimensions are divisible by patch_size
        h_new = (h // patch_size) * patch_size
        w_new = (w // patch_size) * patch_size
        
        mel_spec = mel_spec[:h_new, :w_new]
        
        # Create patches
        patches = []
        for i in range(0, h_new, patch_size):
            for j in range(0, w_new, patch_size):
                patch = mel_spec[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())
        
        return np.array(patches)
    
    def process_batch(self, file_batch, label_value):
        """Process a batch of files to reduce memory usage"""
        batch_data = []
        batch_labels = []
        
        for file_path in file_batch:
            try:
                # Load audio
                audio = self.audio_processor.load_audio(
                    file_path, 
                    max_length=self.config.MAX_AUDIO_LENGTH
                )
                
                if audio is not None:
                    # Convert to mel-spectrogram
                    mel_spec = self.audio_processor.audio_to_melspectrogram(
                        audio, 
                        img_size=224  # Use standard size for UNETR
                    )
                    
                    # Convert to grayscale for UNETR (take one channel)
                    if len(mel_spec.shape) == 3:
                        mel_spec = mel_spec[:, :, 0]
                    
                    # Create patches
                    patches = self.create_patches(mel_spec, self.config.UNETR_PATCH_SIZE)
                    
                    # Normalize patches
                    patches = patches.astype(np.float32) / 255.0
                    
                    batch_data.append(patches)
                    batch_labels.append(label_value)
                    
                # Clear audio variable to save memory
                del audio
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        return batch_data, batch_labels
    
    def save_batch_data(self, split, batch_idx, batch_data, batch_labels):
        """Save batch data incrementally"""
        output_path = self.config.PREPROCESSED_DIR / "unetr" / f"{split}_batch_{batch_idx}.pkl"
        batch_info = {
            'data': batch_data,
            'labels': batch_labels
        }
        with open(output_path, 'wb') as f:
            pickle.dump(batch_info, f)
        
        # Clear batch data from memory
        del batch_data, batch_labels, batch_info
        gc.collect()
        
        return output_path
    
    def preprocess_dataset(self):
        """Preprocess entire dataset for UNETR with memory-efficient batch processing"""
        print("Starting UNETR preprocessing...")
        
        # Get dataset structure
        dataset_structure = self.audio_processor.get_dataset_paths()
        
        batch_metadata = {
            'training': [],
            'validation': [], 
            'testing': []
        }
        
        for split in ['training', 'validation', 'testing']:
            print(f"\nProcessing {split} split...")
            batch_idx = 0
            
            for label, file_paths in dataset_structure[split].items():
                label_value = 0 if label == 'real' else 1  # 0: real, 1: fake
                
                print(f"Processing {len(file_paths)} {label} files...")
                
                # Process files in batches
                for i in range(0, len(file_paths), self.batch_size):
                    batch_files = file_paths[i:i + self.batch_size]
                    
                    try:
                        # Process batch
                        batch_data, batch_labels = self.process_batch(batch_files, label_value)
                        
                        if len(batch_data) > 0:  # Only save if batch has data
                            # Save batch incrementally
                            output_path = self.save_batch_data(split, batch_idx, batch_data, batch_labels)
                            
                            # Store metadata
                            batch_metadata[split].append({
                                'batch_idx': batch_idx,
                                'file_path': str(output_path),
                                'num_samples': len(batch_data),
                                'label': label,
                                'label_value': label_value
                            })
                            
                            print(f"Saved batch {batch_idx} with {len(batch_data)} samples to {output_path}")
                            batch_idx += 1
                        
                        # Force garbage collection
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue
            
            print(f"Completed {split} split - {batch_idx} batches saved")
        
        # Save batch metadata
        metadata_path = self.config.PREPROCESSED_DIR / "unetr" / "batch_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(batch_metadata, f)
        
        # Calculate and save dataset statistics
        total_samples = {split: sum(batch['num_samples'] for batch in batches) 
                        for split, batches in batch_metadata.items()}
        
        stats = {
            'training_samples': total_samples['training'],
            'validation_samples': total_samples['validation'],
            'testing_samples': total_samples['testing'],
            'patch_size': self.config.UNETR_PATCH_SIZE,
            'classes': ['real', 'fake'],
            'batch_size': self.batch_size,
            'batch_metadata': batch_metadata
        }
        
        stats_path = self.config.PREPROCESSED_DIR / "unetr" / "dataset_stats.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"\nUNETR preprocessing completed!")
        print(f"Dataset statistics saved to {stats_path}")
        print(f"Batch metadata saved to {metadata_path}")
        print(f"Training samples: {stats['training_samples']}")
        print(f"Validation samples: {stats['validation_samples']}")
        print(f"Testing samples: {stats['testing_samples']}")
        
        return stats

if __name__ == "__main__":
    # Use smaller batch size to reduce memory usage
    preprocessor = UNETRPreprocessor(batch_size=8)
    preprocessor.preprocess_dataset()