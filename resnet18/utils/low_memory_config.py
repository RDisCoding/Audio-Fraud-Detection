"""
Memory-optimized configuration for audio fraud detection preprocessing
Use this configuration when running on systems with limited memory
"""
import os
from pathlib import Path

class LowMemoryConfig:
    # Base paths
    BASE_DIR = Path("D:/Audio Fraud Detection System/audio_fraud_detection")
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
    
    # Audio processing parameters (optimized for low memory)
    SAMPLE_RATE = 16000
    MAX_AUDIO_LENGTH = 5  # Reduced from 10 to 5 seconds
    N_MELS = 64  # Reduced from 128 to 64
    N_FFT = 1024  # Reduced from 2048 to 1024
    HOP_LENGTH = 256  # Reduced from 512 to 256
    
    # Training parameters (optimized for low memory)
    BATCH_SIZE = 4  # Reduced from 16 to 4
    LEARNING_RATE = 0.001
    EPOCHS = 50
    PATIENCE = 10
    
    # Model specific parameters (reduced sizes)
    RESNET_IMG_SIZE = 128  # Reduced from 224 to 128
    UNETR_PATCH_SIZE = 8   # Reduced from 16 to 8
    MOBILENET_IMG_SIZE = 128  # Reduced from 224 to 128
    
    # Memory management parameters
    PREPROCESSING_BATCH_SIZE = 4  # Very small batch size for preprocessing
    MAX_MEMORY_PERCENT = 80  # Maximum memory usage before forcing cleanup
    GARBAGE_COLLECTION_FREQUENCY = 10  # Force GC every N batches
    
    # Data type optimizations
    AUDIO_DTYPE = 'float32'  # Use float32 instead of float64
    IMAGE_DTYPE = 'uint8'    # Use uint8 for images
    
    # XGBoost parameters (reduced complexity)
    XGBOOST_PARAMS = {
        'max_depth': 4,  # Reduced from 6 to 4
        'learning_rate': 0.1,
        'n_estimators': 50,  # Reduced from 100 to 50
        'random_state': 42
    }
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.PREPROCESSED_DIR,
            cls.PREPROCESSED_DIR / "resnet",
            cls.PREPROCESSED_DIR / "mobilenet", 
            cls.PREPROCESSED_DIR / "unetr",
            cls.FEATURES_DIR,
            cls.MODELS_DIR,
            cls.MODELS_DIR / "resnet",
            cls.MODELS_DIR / "mobilenet",
            cls.MODELS_DIR / "unetr",
            cls.SAVED_MODELS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def setup_memory_optimization(cls):
        """Setup environment variables for memory optimization"""
        # Limit numpy/scipy threading to reduce memory overhead
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1' 
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        # Set librosa cache to minimal
        os.environ['LIBROSA_CACHE_LEVEL'] = '10'
        
        # Disable TensorFlow memory growth (if using TF)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        print("Memory optimization settings applied")
