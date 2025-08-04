# Audio Fraud Detection System - Execution Guide

## Prerequisites
1. Ensure Python 3.8+ is installed
2. Ensure the FoR dataset is placed at: D:/Audio Fraud Detection System/for-norm/for-norm
3. Activate your virtual environment: audio_fraud_env\Scripts\activate

## Step-by-Step Execution

### 1. Data Preprocessing (Run these in order)
```bash
cd "D:/Audio Fraud Detection System/audio_fraud_detection"
python scripts/preprocessing/preprocess_resnet.py
python scripts/preprocessing/preprocess_unetr.py  
python scripts/preprocessing/preprocess_mobilenet.py
```

### 2. Individual Model Training (Run these separately to manage memory)
```bash
# Train ResNet (estimated 4-6 hours)
python scripts/training/train_resnet.py

# Train UNETR (estimated 6-8 hours) 
python scripts/training/train_unetr.py

# Train MobileNet (estimated 2-3 hours)
python scripts/training/train_mobilenet.py
```

### 3. Fusion Model Training
```bash
# Extract features and train XGBoost fusion model
python scripts/training/train_fusion.py
```

### 4. Model Evaluation
```bash
# Compare all models and generate reports
python scripts/evaluation/evaluate_models.py
```

## Expected Timeline
- Data Preprocessing: 2-4 hours
- Individual Model Training: 12-17 hours total
- Fusion Training: 1-2 hours
- Evaluation: 30 minutes
- **Total: 15-23 hours**

## Monitoring Progress
- TensorBoard logs are saved in the `logs/` directory
- Model checkpoints are saved every 10 epochs
- Best models are automatically saved when validation accuracy improves

## Troubleshooting
- If you run out of GPU memory, reduce BATCH_SIZE in utils/config.py
- If preprocessing takes too long, consider using a subset of data for initial testing
- Check the logs/ directory for detailed training logs

## Results
- Final model comparison will be saved in results/
- Best performing model information will be displayed at the end
- Expected accuracy: 95%+ with the fusion approach
