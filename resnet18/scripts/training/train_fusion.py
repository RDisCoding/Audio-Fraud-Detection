import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from utils.audio_processing import AudioProcessor

# Import model architectures
sys.path.append(str(project_root / "scripts" / "training"))
from train_resnet import ResNetTrainer
from train_unetr import UNETRTrainer, AudioTransformer
from train_mobilenet import MobileNetTrainer

class FusionModelTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load trained models
        self.resnet_model = None
        self.unetr_model = None
        self.mobilenet_model = None
        self.xgb_model = None
        
    def load_trained_models(self):
        """Load all trained individual models"""
        print("Loading trained models...")
        
        # Load ResNet
        try:
            resnet_path = self.config.SAVED_MODELS_DIR / "resnet_best.pth"
            resnet_checkpoint = torch.load(resnet_path, map_location=self.device)
            
            resnet_trainer = ResNetTrainer()
            self.resnet_model = resnet_trainer._build_model()
            self.resnet_model.load_state_dict(resnet_checkpoint['model_state_dict'])
            self.resnet_model.eval()
            print("✓ ResNet model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading ResNet model: {e}")
            return False
        
        # Load UNETR
        try:
            unetr_path = self.config.SAVED_MODELS_DIR / "unetr_best.pth"
            unetr_checkpoint = torch.load(unetr_path, map_location=self.device)
            
            # Get patch dimension from saved stats
            dataset_stats = unetr_checkpoint['dataset_stats']
            patch_dim = dataset_stats['patch_dimension']
            
            self.unetr_model = AudioTransformer(
                patch_dim=patch_dim,
                d_model=256,
                nhead=8,
                num_layers=4,
                num_classes=2
            ).to(self.device)
            self.unetr_model.load_state_dict(unetr_checkpoint['model_state_dict'])
            self.unetr_model.eval()
            print("✓ UNETR model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading UNETR model: {e}")
            return False
        
        # Load MobileNet
        try:
            mobilenet_path = self.config.SAVED_MODELS_DIR / "mobilenet_best.pth"
            mobilenet_checkpoint = torch.load(mobilenet_path, map_location=self.device)
            
            mobilenet_trainer = MobileNetTrainer()
            self.mobilenet_model = mobilenet_trainer._build_model()
            self.mobilenet_model.load_state_dict(mobilenet_checkpoint['model_state_dict'])
            self.mobilenet_model.eval()
            print("✓ MobileNet model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading MobileNet model: {e}")
            return False
        
        return True
    
    def extract_features(self, split='training'):
        """Extract features from all three models for a given split"""
        print(f"Extracting features for {split} split...")
        
        features_dict = {
            'resnet': [],
            'unetr': [],
            'mobilenet': [],
            'labels': []
        }
        
        # Load preprocessed data for each model
        resnet_path = self.config.PREPROCESSED_DIR / "resnet" / f"{split}_data.pkl"
        unetr_path = self.config.PREPROCESSED_DIR / "unetr" / f"{split}_data.pkl"
        mobilenet_path = self.config.PREPROCESSED_DIR / "mobilenet" / f"{split}_data.pkl"
        
        with open(resnet_path, 'rb') as f:
            resnet_data = pickle.load(f)
        with open(unetr_path, 'rb') as f:
            unetr_data = pickle.load(f)
        with open(mobilenet_path, 'rb') as f:
            mobilenet_data = pickle.load(f)
        
        # Ensure all datasets have the same number of samples
        min_samples = min(len(resnet_data['labels']), 
                         len(unetr_data['labels']), 
                         len(mobilenet_data['labels']))
        
        print(f"Processing {min_samples} samples...")
        
        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 8
            
            for i in tqdm(range(0, min_samples, batch_size), desc=f"Extracting {split} features"):
                end_idx = min(i + batch_size, min_samples)
                
                # ResNet features
                resnet_batch = torch.FloatTensor(resnet_data['data'][i:end_idx]).to(self.device)
                resnet_output = self.resnet_model(resnet_batch)
                resnet_features = torch.softmax(resnet_output, dim=1).cpu().numpy()
                features_dict['resnet'].extend(resnet_features)
                
                # UNETR features
                unetr_batch = torch.FloatTensor(unetr_data['data'][i:end_idx]).to(self.device)
                unetr_output = self.unetr_model(unetr_batch)
                unetr_features = torch.softmax(unetr_output, dim=1).cpu().numpy()
                features_dict['unetr'].extend(unetr_features)
                
                # MobileNet features
                mobilenet_batch = torch.FloatTensor(mobilenet_data['data'][i:end_idx]).to(self.device)
                mobilenet_output = self.mobilenet_model(mobilenet_batch)
                mobilenet_features = torch.softmax(mobilenet_output, dim=1).cpu().numpy()
                features_dict['mobilenet'].extend(mobilenet_features)
                
                # Labels (use ResNet labels as reference)
                if i == 0:  # Only add labels once
                    features_dict['labels'] = resnet_data['labels'][:min_samples].tolist()
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Convert to numpy arrays
        features_dict['resnet'] = np.array(features_dict['resnet'])
        features_dict['unetr'] = np.array(features_dict['unetr'])
        features_dict['mobilenet'] = np.array(features_dict['mobilenet'])
        features_dict['labels'] = np.array(features_dict['labels'])
        
        # Combine all features
        combined_features = np.hstack([
            features_dict['resnet'],
            features_dict['unetr'],
            features_dict['mobilenet']
        ])
        
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Labels shape: {features_dict['labels'].shape}")
        
        # Save extracted features
        features_output_path = self.config.FEATURES_DIR / f"{split}_fusion_features.pkl"
        with open(features_output_path, 'wb') as f:
            pickle.dump({
                'features': combined_features,
                'labels': features_dict['labels'],
                'individual_features': features_dict
            }, f)
        
        print(f"Features saved to {features_output_path}")
        return combined_features, features_dict['labels']
    
    def train_xgboost(self):
        """Train XGBoost classifier on extracted features"""
        print("Training XGBoost fusion classifier...")
        
        # Extract features for training and validation
        train_features, train_labels = self.extract_features('training')
        val_features, val_labels = self.extract_features('validation')
        
        # Train XGBoost
        self.xgb_model = xgb.XGBClassifier(**self.config.XGBOOST_PARAMS)
        
        # Training with validation set for early stopping
        self.xgb_model.fit(
            train_features, train_labels,
            eval_set=[(val_features, val_labels)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Evaluate on validation set
        val_predictions = self.xgb_model.predict(val_features)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Classification Report:")
        print(classification_report(val_labels, val_predictions, target_names=['Real', 'Fake']))
        
        # Save XGBoost model
        xgb_model_path = self.config.SAVED_MODELS_DIR / "xgboost_fusion.pkl"
        joblib.dump(self.xgb_model, xgb_model_path)
        print(f"XGBoost model saved to {xgb_model_path}")
        
        return val_accuracy
    
    def evaluate_on_test(self):
        """Evaluate the fusion model on test set"""
        print("Evaluating fusion model on test set...")
        
        # Extract test features
        test_features, test_labels = self.extract_features('testing')
        
        # Load XGBoost model if not already loaded
        if self.xgb_model is None:
            xgb_model_path = self.config.SAVED_MODELS_DIR / "xgboost_fusion.pkl"
            self.xgb_model = joblib.load(xgb_model_path)
        
        # Make predictions
        test_predictions = self.xgb_model.predict(test_features)
        test_probabilities = self.xgb_model.predict_proba(test_features)
        
        # Calculate metrics
        test_accuracy = accuracy_score(test_labels, test_predictions)
        
        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        print(classification_report(test_labels, test_predictions, target_names=['Real', 'Fake']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(test_labels, test_predictions))
        
        # Save results
        results = {
            'test_accuracy': test_accuracy,
            'test_predictions': test_predictions,
            'test_probabilities': test_probabilities,
            'test_labels': test_labels,
            'classification_report': classification_report(test_labels, test_predictions, target_names=['Real', 'Fake'], output_dict=True),
            'confusion_matrix': confusion_matrix(test_labels, test_predictions)
        }
        
        results_path = self.config.RESULTS_DIR / "fusion_model_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {results_path}")
        return test_accuracy
    
    def train_complete_fusion_model(self):
        """Complete fusion model training pipeline"""
        print("Starting fusion model training pipeline...")
        
        # Step 1: Load trained individual models
        if not self.load_trained_models():
            print("Failed to load individual models. Please train them first.")
            return None
        
        # Step 2: Train XGBoost fusion classifier
        val_accuracy = self.train_xgboost()
        
        # Step 3: Evaluate on test set
        test_accuracy = self.evaluate_on_test()
        
        print(f"\n=== FUSION MODEL TRAINING COMPLETED ===")
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_accuracy

if __name__ == "__main__":
    trainer = FusionModelTrainer()
    trainer.train_complete_fusion_model()