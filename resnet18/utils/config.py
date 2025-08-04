import os
import gc
import psutil
import torch
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path("D:/Audio Fraud Detection System/audio_fraud_detection_fusion/audio_fraud_detection")
    DATASET_PATH = Path("D:/Audio Fraud Detection System/for-norm/for-norm")
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    PREPROCESSED_DIR = DATA_DIR / "preprocessed"
    FEATURES_DIR = DATA_DIR / "features"
    
    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
    
    # Results and logs
    LOGS_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Audio processing parameters
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 10  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 50
    PATIENCE = 10
    
    # Model specific parameters
    RESNET_IMG_SIZE = 224
    UNETR_PATCH_SIZE = 16
    MOBILENET_IMG_SIZE = 224
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'random_state': 42
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.PREPROCESSED_DIR / "resnet",
            cls.PREPROCESSED_DIR / "unetr", 
            cls.PREPROCESSED_DIR / "mobilenet",
            cls.FEATURES_DIR,
            cls.MODELS_DIR / "resnet",
            cls.MODELS_DIR / "unetr",
            cls.MODELS_DIR / "mobilenet",
            cls.SAVED_MODELS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("All directories created successfully!")
    
    @classmethod
    def optimize_for_memory(cls):
        """Setup environment variables for memory optimization"""
        # Limit numpy/scipy threading to reduce memory overhead
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2' 
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['NUMEXPR_NUM_THREADS'] = '2'
        
        # Set librosa cache to minimal
        os.environ['LIBROSA_CACHE_LEVEL'] = '10'
        
        # Disable TensorFlow memory growth (if using TF)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        print("Memory optimization settings applied")
    
    @classmethod
    def get_memory_info(cls):
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    @classmethod
    def clear_gpu_cache(cls):
        """Clear GPU cache if using CUDA"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    @classmethod
    def get_dynamic_batch_size(cls):
        """Get batch size based on available memory"""
        memory_info = cls.get_memory_info()
        available_gb = memory_info['available_gb']
        
        # Adjust batch size based on available memory
        if available_gb < 4:
            return 4
        elif available_gb < 8:
            return 8
        elif available_gb < 16:
            return cls.BATCH_SIZE
        else:
            return min(cls.BATCH_SIZE * 2, 32)  # Cap at 32