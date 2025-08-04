import os
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config

class ModelEvaluator:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate_individual_model(self, model_name):
        """Evaluate individual model performance"""
        print(f"Evaluating {model_name} model...")
        
        # Load test data
        test_data_path = self.config.PREPROCESSED_DIR / model_name / "testing_data.pkl"
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # Load model
        model_path = self.config.SAVED_MODELS_DIR / f"{model_name}_best.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load appropriate model architecture
        if model_name == 'resnet':
            from train_resnet import ResNetTrainer
            trainer = ResNetTrainer()
            model = trainer._build_model()
        elif model_name == 'unetr':
            from train_unetr import AudioTransformer
            dataset_stats = checkpoint['dataset_stats']
            model = AudioTransformer(
                patch_dim=dataset_stats['patch_dimension'],
                d_model=256,
                nhead=8,
                num_layers=4,
                num_classes=2
            ).to(self.device)
        elif model_name == 'mobilenet':
            from train_mobilenet import MobileNetTrainer
            trainer = MobileNetTrainer()
            model = trainer._build_model()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare test data
        test_X = torch.FloatTensor(test_data['data'])
        test_y = test_data['labels']
        
        # Make predictions
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            batch_size = 16
            for i in range(0, len(test_X), batch_size):
                batch = test_X[i:i+batch_size].to(self.device)
                output = model(batch)
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(test_y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_y, predictions, average='weighted')
        auc = roc_auc_score(test_y, probabilities[:, 1])
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': test_y
        }
        
        print(f"{model_name.upper()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def evaluate_fusion_model(self):
        """Evaluate fusion model performance"""
        print("Evaluating fusion model...")
        
        # Load fusion model results
        results_path = self.config.RESULTS_DIR / "fusion_model_results.pkl"
        with open(results_path, 'rb') as f:
            fusion_results = pickle.load(f)
        
        test_y = fusion_results['test_labels']
        predictions = fusion_results['test_predictions']
        probabilities = fusion_results['test_probabilities']
        
        # Calculate metrics
        accuracy = accuracy_score(test_y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_y, predictions, average='weighted')
        auc = roc_auc_score(test_y, probabilities[:, 1])
        
        results = {
            'model_name': 'fusion',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': test_y
        }
        
        print(f"FUSION MODEL Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return results
    
    def compare_all_models(self):
        """Compare all models and generate comprehensive evaluation"""
        print("Comparing all models...")
        
        all_results = []
        
        # Evaluate individual models
        for model_name in ['resnet', 'unetr', 'mobilenet']:
            try:
                results = self.evaluate_individual_model(model_name)
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        # Evaluate fusion model
        try:
            fusion_results = self.evaluate_fusion_model()
            all_results.append(fusion_results)
        except Exception as e:
            print(f"Error evaluating fusion model: {e}")
        
        # Create comparison plots
        self.create_comparison_plots(all_results)
        
        # Create summary table
        self.create_summary_table(all_results)
        
        return all_results
    
    def create_comparison_plots(self, results_list):
        """Create comparison plots for all models"""
        print("Creating comparison plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        model_names = [r['model_name'].upper() for r in results_list]
        accuracies = [r['accuracy'] for r in results_list]
        precisions = [r['precision'] for r in results_list]
        recalls = [r['recall'] for r in results_list]
        f1_scores = [r['f1_score'] for r in results_list]
        aucs = [r['auc'] for r in results_list]
        
        # Plot 1: Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 2: Precision, Recall, F1 comparison
        x = np.arange(len(model_names))
        width = 0.25
        axes[0, 1].bar(x - width, precisions, width, label='Precision', alpha=0.8)
        axes[0, 1].bar(x, recalls, width, label='Recall', alpha=0.8)
        axes[0, 1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[0, 1].set_title('Precision, Recall, and F1-Score Comparison')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: AUC comparison
        axes[1, 0].bar(model_names, aucs, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
        axes[1, 0].set_title('AUC Score Comparison')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(aucs):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Plot 4: ROC Curves
        for result in results_list:
            fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'][:, 1])
            axes[1, 1].plot(fpr, tpr, label=f"{result['model_name'].upper()} (AUC = {result['auc']:.3f})")
        
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curves Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.config.RESULTS_DIR / "model_comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plots saved to {plot_path}")
    
    def create_summary_table(self, results_list):
        """Create and save summary table"""
        print("Creating summary table...")
        
        # Create summary data
        summary_data = {
            'Model': [r['model_name'].upper() for r in results_list],
            'Accuracy': [f"{r['accuracy']:.4f}" for r in results_list],
            'Precision': [f"{r['precision']:.4f}" for r in results_list],
            'Recall': [f"{r['recall']:.4f}" for r in results_list],
            'F1-Score': [f"{r['f1_score']:.4f}" for r in results_list],
            'AUC': [f"{r['auc']:.4f}" for r in results_list]
        }
        
        # Create and save table
        import pandas as pd
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = self.config.RESULTS_DIR / "model_comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as pickle for later use
        pickle_path = self.config.RESULTS_DIR / "all_model_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_list, f)
        
        print("Summary Table:")
        print(df.to_string(index=False))
        print(f"\nSummary saved to {csv_path}")
        print(f"Complete results saved to {pickle_path}")
        
        # Find best model
        best_accuracy_idx = np.argmax([r['accuracy'] for r in results_list])
        best_model = results_list[best_accuracy_idx]
        
        print(f"\n=== BEST PERFORMING MODEL ===")
        print(f"Model: {best_model['model_name'].upper()}")
        print(f"Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
        print(f"AUC: {best_model['auc']:.4f}")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.compare_all_models()