import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import pickle
import gc
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.low_memory_config import LowMemoryConfig
from utils.audio_processing import AudioProcessor
from utils.memory_utils import MemoryManager, cleanup_variables

class LowMemoryResNetPreprocessor:
    def __init__(self, batch_size=None):
        self.config = LowMemoryConfig()
        self.config.setup_memory_optimization()
        self.audio_processor = AudioProcessor()
        self.config.create_directories()
        self.memory_manager = MemoryManager()
        
        # Use very small batch size for low memory systems
        self.batch_size = batch_size or self.config.PREPROCESSING_BATCH_SIZE
        
        print(f"Initialized with batch size: {self.batch_size}")
        print(f"Current memory usage: {self.memory_manager.get_memory_usage():.2f} MB")
        
    def process_single_file(self, file_path, label_value):
        """Process a single file to minimize memory usage"""
        try:
            # Load audio
            audio = self.audio_processor.load_audio(
                file_path, 
                max_length=self.config.MAX_AUDIO_LENGTH
            )
            
            if audio is not None:
                # Convert to mel-spectrogram image
                mel_image = self.audio_processor.audio_to_melspectrogram(
                    audio, 
                    img_size=self.config.RESNET_IMG_SIZE
                )
                
                if mel_image is not None:
                    # Normalize for ResNet (ImageNet normalization)
                    mel_image = mel_image.astype(np.float32) / 255.0
                    mel_image = (mel_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    
                    # Transpose to (C, H, W) format
                    mel_image = np.transpose(mel_image, (2, 0, 1))
                    
                    # Clear audio variable immediately
                    cleanup_variables(audio)
                    
                    return mel_image, label_value
            
            # Clean up if processing failed
            cleanup_variables(audio)
            return None, None
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None
    
    def process_batch(self, file_batch, label_value):
        """Process a batch of files with aggressive memory management"""
        batch_data = []
        batch_labels = []
        
        for idx, file_path in enumerate(file_batch):
            # Check memory usage every few files
            if idx % 2 == 0:
                self.memory_manager.check_memory_limit(self.config.MAX_MEMORY_PERCENT)
            
            mel_image, label = self.process_single_file(file_path, label_value)
            
            if mel_image is not None:
                batch_data.append(mel_image)
                batch_labels.append(label)
            
            # Force garbage collection every few files
            if idx % 2 == 1:
                gc.collect()
        
        if batch_data:
            return np.array(batch_data, dtype=self.config.IMAGE_DTYPE), np.array(batch_labels)
        else:
            return np.array([]), np.array([])
    
    def save_batch_data(self, split, batch_idx, batch_data, batch_labels):
        """Save batch data incrementally with compression"""
        output_path = self.config.PREPROCESSED_DIR / "resnet" / f"{split}_batch_{batch_idx}.pkl"
        
        # Use compression to save disk space and potentially memory
        batch_info = {
            'data': batch_data,
            'labels': batch_labels
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(batch_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Immediate cleanup
        cleanup_variables(batch_data, batch_labels, batch_info)
        
        return output_path
        
    def preprocess_dataset(self):
        """Preprocess entire dataset with extreme memory efficiency"""
        print("Starting ResNet preprocessing with low memory configuration...")
        print(f"Target image size: {self.config.RESNET_IMG_SIZE}x{self.config.RESNET_IMG_SIZE}")
        print(f"Audio length: {self.config.MAX_AUDIO_LENGTH} seconds")
        print(f"Batch size: {self.batch_size}")
        
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
                print(f"Memory usage before {label}: {self.memory_manager.get_memory_usage():.2f} MB")
                
                # Process files in very small batches
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
                            
                            print(f"Saved batch {batch_idx} with {len(batch_data)} samples")
                            batch_idx += 1
                        
                        # Aggressive memory cleanup
                        self.memory_manager.force_garbage_collection()
                        
                        # Check memory usage
                        current_memory = self.memory_manager.get_memory_usage()
                        if current_memory > 1000:  # If using more than 1GB
                            print(f"High memory usage detected: {current_memory:.2f} MB")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        # Try to recover by forcing cleanup
                        self.memory_manager.force_garbage_collection()
                        continue
            
            print(f"Completed {split} split - {batch_idx} batches saved")
            print(f"Memory usage after {split}: {self.memory_manager.get_memory_usage():.2f} MB")
        
        # Save batch metadata
        metadata_path = self.config.PREPROCESSED_DIR / "resnet" / "batch_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(batch_metadata, f)
        
        # Calculate and save dataset statistics
        total_samples = {split: sum(batch['num_samples'] for batch in batches) 
                        for split, batches in batch_metadata.items()}
        
        stats = {
            'training_samples': total_samples['training'],
            'validation_samples': total_samples['validation'],
            'testing_samples': total_samples['testing'],
            'image_size': self.config.RESNET_IMG_SIZE,
            'channels': 3,
            'classes': ['real', 'fake'],
            'batch_size': self.batch_size,
            'batch_metadata': batch_metadata,
            'memory_optimized': True,
            'max_audio_length': self.config.MAX_AUDIO_LENGTH,
            'n_mels': self.config.N_MELS
        }
        
        stats_path = self.config.PREPROCESSED_DIR / "resnet" / "dataset_stats.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"\nLow-memory ResNet preprocessing completed!")
        print(f"Dataset statistics saved to {stats_path}")
        print(f"Batch metadata saved to {metadata_path}")
        print(f"Training samples: {stats['training_samples']}")
        print(f"Validation samples: {stats['validation_samples']}")
        print(f"Testing samples: {stats['testing_samples']}")
        print(f"Final memory usage: {self.memory_manager.get_memory_usage():.2f} MB")
        
        return stats

if __name__ == "__main__":
    # Use very small batch size for extreme memory efficiency
    preprocessor = LowMemoryResNetPreprocessor(batch_size=2)
    preprocessor.preprocess_dataset()
