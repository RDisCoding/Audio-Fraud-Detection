#!/usr/bin/env python3
"""
Start ResNet training for audio fraud detection
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from scripts.training.train_resnet import ResNetTrainer

def main():
    print("Starting ResNet training for audio fraud detection...")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = ResNetTrainer()
        
        # Start training
        trainer.train()
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
