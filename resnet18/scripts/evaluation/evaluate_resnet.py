#!/usr/bin/env python3
"""
Evaluate trained ResNet model on test data and save results
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config
from scripts.training.train_resnet import MemoryEfficientDataset

class ResNetEvaluator:
    def __init__(self, model_path=None):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the best trained model
        if model_path is None:
            model_path = self.config.SAVED_MODELS_DIR / "resnet_best.pth"
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Evaluation transforms (same as validation)
        self.test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Results directory
        self.results_dir = self.config.RESULTS_DIR / "resnet_evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.results_dir}")
    
    def _load_model(self, model_path):
        """Load the trained ResNet model"""
        print(f"Loading model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Build model architecture (same as training)
        model = models.resnet18(pretrained=False)
        
        # Freeze early layers (same as training)
        for param in model.layer1.parameters():
            param.requires_grad = False
        
        # Same classifier architecture
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded successfully!")
        print(f"Best validation accuracy from training: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        
        return model
    
    def load_test_data(self):
        """Load test data"""
        print("Loading test data...")
        
        data_dir = self.config.PREPROCESSED_DIR / "resnet"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")
        
        # Check for test data files
        test_pattern = "testing_batch_*.pkl"
        test_files = list(data_dir.glob(test_pattern))
        
        if not test_files:
            # If no specific test files, use validation as test
            print("No test batch files found, using validation data as test set")
            test_dataset = MemoryEfficientDataset(data_dir, split='validation', transforms=self.test_transforms)
        else:
            print(f"Found {len(test_files)} test batch files")
            test_dataset = MemoryEfficientDataset(data_dir, split='testing', transforms=self.test_transforms)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,  # Larger batch size for evaluation
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"Test samples: {len(test_dataset)}")
        return test_loader, test_dataset
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        print("Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Evaluating")):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_losses.append(loss.item())
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'targets': np.array(all_targets),
            'average_loss': np.mean(all_losses)
        }
    
    def calculate_metrics(self, results):
        """Calculate comprehensive evaluation metrics"""
        predictions = results['predictions']
        probabilities = results['probabilities']
        targets = results['targets']
        
        # Basic metrics
        accuracy = np.mean(predictions == targets) * 100
        
        # Classification report
        class_names = ['Genuine', 'Fraudulent']
        report = classification_report(
            targets, predictions, 
            target_names=class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # ROC AUC
        fraud_probabilities = probabilities[:, 1]  # Probability of fraud class
        roc_auc = roc_auc_score(targets, fraud_probabilities)
        
        # Average Precision Score
        avg_precision = average_precision_score(targets, fraud_probabilities)
        
        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(targets, fraud_probabilities)
        
        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(targets, fraud_probabilities)
        
        metrics = {
            'test_accuracy': accuracy,
            'test_loss': results['average_loss'],
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thresholds.tolist()},
            'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': pr_thresholds.tolist()}
        }
        
        return metrics
    
    def save_detailed_results(self, results, metrics):
        """Save detailed results to files"""
        print("Saving detailed results...")
        
        # Save raw predictions
        predictions_df = pd.DataFrame({
            'true_label': results['targets'],
            'predicted_label': results['predictions'],
            'genuine_probability': results['probabilities'][:, 0],
            'fraud_probability': results['probabilities'][:, 1],
            'correct_prediction': results['targets'] == results['predictions']
        })
        
        predictions_file = self.results_dir / "detailed_predictions.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Detailed predictions saved to: {predictions_file}")
        
        # Save metrics to JSON
        metrics_file = self.results_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_file}")
        
        # Save summary report
        summary_file = self.results_dir / "evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("ResNet Audio Fraud Detection - Test Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Accuracy: {metrics['test_accuracy']:.4f}%\n")
            f.write(f"Test Loss: {metrics['test_loss']:.6f}\n")
            f.write(f"ROC AUC Score: {metrics['roc_auc']:.6f}\n")
            f.write(f"Average Precision: {metrics['average_precision']:.6f}\n\n")
            
            f.write("Classification Report:\n")
            f.write("-" * 30 + "\n")
            report = metrics['classification_report']
            for class_name in ['Genuine', 'Fraudulent']:
                class_key = '0' if class_name == 'Genuine' else '1'
                if class_key in report:
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {report[class_key]['precision']:.4f}\n")
                    f.write(f"  Recall: {report[class_key]['recall']:.4f}\n")
                    f.write(f"  F1-Score: {report[class_key]['f1-score']:.4f}\n\n")
            
            f.write(f"Macro Average F1: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Average F1: {report['weighted avg']['f1-score']:.4f}\n")
            
        print(f"Summary report saved to: {summary_file}")
    
    def create_visualizations(self, results, metrics):
        """Create and save visualization plots"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Genuine', 'Fraudulent'],
                   yticklabels=['Genuine', 'Fraudulent'])
        plt.title('Confusion Matrix - ResNet Audio Fraud Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ResNet Audio Fraud Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision = metrics['pr_curve']['precision']
        recall = metrics['pr_curve']['recall']
        plt.plot(recall, precision, color='purple', lw=2,
                label=f'PR curve (AP = {metrics["average_precision"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - ResNet Audio Fraud Detection')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Prediction Distribution
        plt.figure(figsize=(12, 4))
        
        # Fraud probability distribution
        plt.subplot(1, 2, 1)
        genuine_probs = results['probabilities'][results['targets'] == 0, 1]
        fraud_probs = results['probabilities'][results['targets'] == 1, 1]
        
        plt.hist(genuine_probs, bins=30, alpha=0.7, label='Genuine Audio', density=True)
        plt.hist(fraud_probs, bins=30, alpha=0.7, label='Fraudulent Audio', density=True)
        plt.xlabel('Fraud Probability')
        plt.ylabel('Density')
        plt.title('Fraud Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy by confidence
        plt.subplot(1, 2, 2)
        max_probs = np.max(results['probabilities'], axis=1)
        confidence_bins = np.linspace(0.5, 1.0, 11)
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(results['predictions'][mask] == results['targets'][mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.plot(confidence_bins[:-1], accuracies, 'o-', linewidth=2, markersize=6)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "prediction_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {self.results_dir}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting ResNet model evaluation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load test data
        test_loader, test_dataset = self.load_test_data()
        
        # Evaluate model
        results = self.evaluate_model(test_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}%")
        print(f"Test Loss: {metrics['test_loss']:.6f}")
        print(f"ROC AUC Score: {metrics['roc_auc']:.6f}")
        print(f"Average Precision: {metrics['average_precision']:.6f}")
        print()
        
        # Print classification report
        report = metrics['classification_report']
        print("Classification Report:")
        print("-" * 30)
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        class_names = ['Genuine', 'Fraudulent']
        for i, class_name in enumerate(class_names):
            class_key = str(i)
            if class_key in report:
                print(f"{class_name:<12} {report[class_key]['precision']:<10.4f} "
                     f"{report[class_key]['recall']:<10.4f} {report[class_key]['f1-score']:<10.4f} "
                     f"{report[class_key]['support']:<10.0f}")
        
        print("-" * 60)
        print(f"{'Macro Avg':<12} {report['macro avg']['precision']:<10.4f} "
             f"{report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f} "
             f"{report['macro avg']['support']:<10.0f}")
        print(f"{'Weighted Avg':<12} {report['weighted avg']['precision']:<10.4f} "
             f"{report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} "
             f"{report['weighted avg']['support']:<10.0f}")
        
        # Save results
        self.save_detailed_results(results, metrics)
        
        # Create visualizations
        self.create_visualizations(results, metrics)
        
        # Clear cache
        test_dataset.clear_cache()
        
        evaluation_time = time.time() - start_time
        print(f"\nEvaluation completed in {evaluation_time/60:.2f} minutes")
        print(f"All results saved to: {self.results_dir}")
        
        return metrics

if __name__ == "__main__":
    evaluator = ResNetEvaluator()
    evaluator.run_evaluation()
