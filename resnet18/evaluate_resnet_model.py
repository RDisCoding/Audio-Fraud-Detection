#!/usr/bin/env python3
"""
Quick script to evaluate the trained ResNet model
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from scripts.evaluation.evaluate_resnet import ResNetEvaluator

def main():
    print("🚀 Starting ResNet Model Evaluation")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        evaluator = ResNetEvaluator()
        
        # Run evaluation
        metrics = evaluator.run_evaluation()
        
        print("\n" + "🎉" * 20)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        print(f"\n📊 Final Results:")
        print(f"   • Test Accuracy: {metrics['test_accuracy']:.2f}%")
        print(f"   • ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   • Average Precision: {metrics['average_precision']:.4f}")
        
        print(f"\n📁 Results saved to:")
        print(f"   • Detailed predictions: results/resnet_evaluation/detailed_predictions.csv")
        print(f"   • Metrics: results/resnet_evaluation/evaluation_metrics.json")
        print(f"   • Summary: results/resnet_evaluation/evaluation_summary.txt")
        print(f"   • Visualizations: results/resnet_evaluation/")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
