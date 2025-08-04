#!/usr/bin/env python3
"""
Test ResNet training setup - run a few training steps to verify everything works
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from scripts.training.train_resnet import ResNetTrainer

def test_training():
    print("Testing ResNet training setup...")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = ResNetTrainer()
        
        # Load data
        print("Loading data...")
        train_loader, val_loader, train_dataset, val_dataset = trainer.load_data()
        
        print(f"✓ Data loaded successfully")
        print(f"✓ Training samples: {len(train_dataset):,}")
        print(f"✓ Validation samples: {len(val_dataset):,}")
        print(f"✓ Batch size: {trainer.batch_size}")
        
        # Test one training step
        print("\nTesting one training batch...")
        trainer.model.train()
        
        # Get one batch
        batch_data, batch_labels = next(iter(train_loader))
        batch_data = batch_data.to(trainer.device)
        batch_labels = batch_labels.to(trainer.device)
        
        # Forward pass
        outputs = trainer.model(batch_data)
        loss = trainer.criterion(outputs, batch_labels)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Batch shape: {batch_data.shape}")
        print(f"✓ Output shape: {outputs.shape}")
        print(f"✓ Loss: {loss.item():.4f}")
        
        # Test one validation step
        print("\nTesting one validation batch...")
        trainer.model.eval()
        
        batch_data, batch_labels = next(iter(val_loader))
        batch_data = batch_data.to(trainer.device)
        batch_labels = batch_labels.to(trainer.device)
        
        with torch.no_grad():
            outputs = trainer.model(batch_data)
            val_loss = trainer.criterion(outputs, batch_labels)
        
        print(f"✓ Validation pass successful")
        print(f"✓ Validation loss: {val_loss.item():.4f}")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Ready to start training.")
        print("Run: python start_resnet_training.py")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    test_training()
