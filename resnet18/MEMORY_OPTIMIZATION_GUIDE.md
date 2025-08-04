# Memory-Optimized Audio Fraud Detection Preprocessing

This guide explains how to use the memory-optimized preprocessing scripts to avoid Out-of-Memory (OOM) errors.

## Overview

The original preprocessing scripts have been modified to handle large datasets on systems with limited memory. The key optimizations include:

1. **Batch Processing**: Files are processed in small batches instead of loading everything into memory
2. **Incremental Saving**: Data is saved immediately after processing each batch
3. **Memory Management**: Aggressive garbage collection and memory monitoring
4. **Reduced Parameters**: Smaller image sizes and audio lengths to reduce memory footprint

## Available Scripts

### 1. Standard Memory-Optimized Scripts
- `preprocess_resnet.py` - ResNet preprocessing with batch processing
- `preprocess_mobilenet.py` - MobileNet preprocessing with batch processing  
- `preprocess_unetr.py` - UNETR preprocessing with batch processing

### 2. Low-Memory Scripts (For Very Limited Memory)
- `preprocess_resnet_low_memory.py` - Extreme memory optimization for ResNet

## Usage

### For Standard Memory-Optimized Preprocessing:

```bash
# ResNet preprocessing (batch size 8)
python scripts/preprocessing/preprocess_resnet.py

# MobileNet preprocessing (batch size 8)
python scripts/preprocessing/preprocess_mobilenet.py

# UNETR preprocessing (batch size 8)
python scripts/preprocessing/preprocess_unetr.py
```

### For Low-Memory Systems:

```bash
# Use the low-memory version with very small batch size
python scripts/preprocessing/preprocess_resnet_low_memory.py
```

## Configuration

### Memory Optimization Settings

The scripts use these optimized settings:

```python
# Standard optimized settings
BATCH_SIZE = 8  # Reduced from 32
MAX_AUDIO_LENGTH = 10  # seconds
RESNET_IMG_SIZE = 224
N_MELS = 128

# Low-memory settings  
BATCH_SIZE = 2-4  # Very small batches
MAX_AUDIO_LENGTH = 5  # Reduced audio length
RESNET_IMG_SIZE = 128  # Smaller images
N_MELS = 64  # Fewer mel bands
```

### Environment Variables for Memory Optimization

The scripts automatically set these environment variables:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
LIBROSA_CACHE_LEVEL=10
```

## Memory Management Features

### 1. Batch Processing
- Files are processed in small batches (2-8 files at a time)
- Each batch is saved immediately after processing
- Memory is cleared between batches

### 2. Memory Monitoring
- Real-time memory usage tracking
- Automatic memory cleanup when usage exceeds 85%
- Warning messages when memory usage is high

### 3. Incremental Saving
- Data is saved as individual batch files instead of large monolithic files
- Metadata tracks all batch files for easy loading during training
- Reduced peak memory usage

## Output Structure

The memory-optimized scripts create this output structure:

```
data/preprocessed/resnet/
├── training_batch_0.pkl
├── training_batch_1.pkl
├── ...
├── validation_batch_0.pkl
├── ...
├── testing_batch_0.pkl
├── ...
├── batch_metadata.pkl  # Maps batch files to labels
└── dataset_stats.pkl   # Overall statistics
```

## Loading Preprocessed Data

To load the batch-processed data during training:

```python
import pickle
from pathlib import Path

# Load metadata
with open('data/preprocessed/resnet/batch_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# Load training batches
training_data = []
training_labels = []

for batch_info in metadata['training']:
    with open(batch_info['file_path'], 'rb') as f:
        batch = pickle.load(f)
        training_data.append(batch['data'])
        training_labels.append(batch['labels'])
```

## Troubleshooting

### Still Getting OOM Errors?

1. **Reduce batch size further**:
   ```python
   preprocessor = ResNetPreprocessor(batch_size=1)
   ```

2. **Use low-memory configuration**:
   ```python
   from utils.low_memory_config import LowMemoryConfig
   config = LowMemoryConfig()
   ```

3. **Reduce audio parameters**:
   - Decrease `MAX_AUDIO_LENGTH` from 10 to 5 seconds
   - Reduce `N_MELS` from 128 to 64
   - Reduce image size from 224 to 128

4. **Close other applications** to free up system memory

5. **Use the low-memory preprocessor**:
   ```bash
   python scripts/preprocessing/preprocess_resnet_low_memory.py
   ```

### Memory Usage Guidelines

- **8GB RAM**: Use batch_size=4, standard settings
- **4GB RAM**: Use batch_size=2, low-memory settings  
- **2GB RAM**: Use batch_size=1, very low-memory settings

### Monitoring Memory Usage

The scripts will print memory usage information:

```
Initial memory usage: 245.32 MB
Processing 150 real files...
Memory usage before real: 245.32 MB
Saved batch 0 with 4 samples
Memory usage after training: 287.45 MB
```

## Performance Considerations

### Trade-offs:
- **Memory Usage**: Significantly reduced (90%+ reduction)
- **Processing Time**: Slightly increased due to frequent I/O operations
- **Disk Usage**: Same total usage, but spread across many small files

### Benefits:
- Can process large datasets on limited memory systems
- More reliable processing with automatic error recovery
- Better monitoring and debugging capabilities
- Scalable to very large datasets

## Next Steps

After preprocessing with these memory-optimized scripts:

1. Update your training scripts to load batch-processed data
2. Use smaller batch sizes during training to maintain memory efficiency
3. Consider using gradient accumulation if you need effective larger batch sizes
4. Monitor memory usage during training and adjust parameters as needed
