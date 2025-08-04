import sys
from pathlib import Path
sys.path.append('.')
from scripts.training.train_resnet import ResNetTrainer
trainer = ResNetTrainer()
print('Checking data loading...')
try:
    train_loader, val_loader, train_dataset, val_dataset = trainer.load_data()
    print('Data loading successful!')
    print(f'Sample data shape: {next(iter(train_loader))[0].shape}')
    print(f'Sample label shape: {next(iter(train_loader))[1].shape}')
except Exception as e:
    print(f'Error: {e}')