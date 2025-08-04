import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import gc

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that loads batch files on-demand with data augmentation"""
    def __init__(self, data_dir, split='training', transforms=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        
        # Find all batch files for this split
        pattern = f"{split}_batch_*.pkl"
        self.batch_files = sorted(list(self.data_dir.glob(pattern)))
        
        if not self.batch_files:
            raise FileNotFoundError(f"No {pattern} files found in {data_dir}")
        
        print(f"Found {len(self.batch_files)} batch files for {split}")
        
        # Load metadata from all batches to get total length
        self.batch_sizes = []
        self.cumulative_sizes = [0]
        
        for batch_file in self.batch_files:
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
                batch_size = len(data['labels'])
                self.batch_sizes.append(batch_size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + batch_size)
        
        self.length = self.cumulative_sizes[-1]
        print(f"Total {split} samples: {self.length}")
        
        # Cache for current batch
        self._cached_batch_idx = -1
        self._cached_data = None
        self._cached_labels = None
    
    def __len__(self):
        return self.length
    
    def _load_batch_if_needed(self, batch_idx):
        """Load batch only when needed"""
        if self._cached_batch_idx != batch_idx:
            batch_file = self.batch_files[batch_idx]
            with open(batch_file, 'rb') as f:
                data = pickle.load(f)
                self._cached_data = data['data']
                self._cached_labels = data['labels']
                self._cached_batch_idx = batch_idx
    
    def __getitem__(self, idx):
        # Find which batch this index belongs to
        batch_idx = 0
        for i, cumulative_size in enumerate(self.cumulative_sizes[1:]):
            if idx < cumulative_size:
                batch_idx = i
                break
        
        # Load the batch if needed
        self._load_batch_if_needed(batch_idx)
        
        # Get local index within the batch
        local_idx = idx - self.cumulative_sizes[batch_idx]
        
        # Convert to tensor
        data = torch.FloatTensor(self._cached_data[local_idx])
        
        # Apply transforms if provided (for training data)
        if self.transforms:
            # Convert from tensor to numpy for PIL transforms
            data_np = (data.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            data = self.transforms(data_np)
        
        label = torch.LongTensor([self._cached_labels[local_idx]])[0]
        
        return data, label
    
    def clear_cache(self):
        """Clear cached data to free memory"""
        self._cached_data = None
        self._cached_labels = None
        self._cached_batch_idx = -1
        gc.collect()

class ResNetTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Memory optimization
        self.config.optimize_for_memory()
        
        # Dynamic batch size based on available memory
        self.batch_size = self.config.get_dynamic_batch_size()
        print(f"Using batch size: {self.batch_size}")
        
        # Initialize model
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization
        
        # Reduced learning rate and added weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE * 0.5,  # Reduced LR
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # More aggressive learning rate scheduling
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.EPOCHS,
            eta_min=1e-6
        )
        
        # Setup logging
        self.writer = SummaryWriter(self.config.LOGS_DIR / 'resnet')
        
        # Add data augmentation transforms
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _build_model(self):
        """Build ResNet model with better regularization"""
        # Use pretrained ResNet18
        model = models.resnet18(pretrained=True)
        
        # Freeze early layers to prevent overfitting
        for param in model.layer1.parameters():
            param.requires_grad = False
        
        # More sophisticated classifier with dropout and batch norm
        model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Increased dropout
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
        
        return model.to(self.device)
    
    def load_data(self):
        """Load preprocessed data with augmentation"""
        print("Loading preprocessed data...")
        
        data_dir = self.config.PREPROCESSED_DIR / "resnet"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")
        
        # Create datasets with appropriate transforms
        train_dataset = MemoryEfficientDataset(data_dir, split='training', transforms=self.train_transforms)
        val_dataset = MemoryEfficientDataset(data_dir, split='validation', transforms=self.val_transforms)
        
        # Create dataloaders with memory optimization
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # Reduced to avoid memory issues
            pin_memory=False,  # Disabled for memory conservation
            drop_last=True  # Avoid variable batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    def train_epoch(self, dataloader, dataset):
        """Train for one epoch with mixup augmentation"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup augmentation occasionally
            if np.random.random() < 0.2:  # 20% chance
                data, target_a, target_b, lam = self.mixup_data(data, target, alpha=0.2)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.mixup_criterion(output, target_a, target_b, lam)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Clear cache every 20 batches
            if batch_idx % 20 == 0:
                self.config.clear_gpu_cache()
                gc.collect()
        
        # Clear dataset cache
        dataset.clear_cache()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixup loss function"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def validate(self, dataloader, dataset):
        """Validate the model with memory management"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Validating")):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    self.config.clear_gpu_cache()
        
        # Clear dataset cache
        dataset.clear_cache()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self):
        """Full training loop with better overfitting prevention"""
        print("Starting ResNet training...")
        
        # Print memory info
        memory_info = self.config.get_memory_info()
        print(f"Available memory: {memory_info['available_gb']:.2f} GB")
        
        # Load data
        train_loader, val_loader, train_dataset, val_dataset = self.load_data()
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        # Track overfitting
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, train_dataset)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, val_dataset)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track losses for overfitting detection
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Calculate and log overfitting metrics
            if len(train_losses) >= 5:
                recent_train_loss = np.mean(train_losses[-5:])
                recent_val_loss = np.mean(val_losses[-5:])
                overfitting_ratio = recent_val_loss / recent_train_loss if recent_train_loss > 0 else 1.0
                self.writer.add_scalar('Overfitting/Loss_Ratio', overfitting_ratio, epoch)
                
                # Warning if overfitting detected
                if overfitting_ratio > 1.2:
                    print(f"⚠️  Overfitting detected! Val/Train loss ratio: {overfitting_ratio:.3f}")
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Print memory usage
            memory_info = self.config.get_memory_info()
            print(f"Memory usage: {memory_info['percent_used']:.1f}%")
            
            # More conservative model saving (only if val loss also improves)
            if val_acc > best_val_acc and (len(val_losses) < 2 or val_loss <= min(val_losses[:-1])):
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                model_path = self.config.SAVED_MODELS_DIR / "resnet_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'config': self.config
                }, model_path)
                
                print(f"✅ New best model saved! Val Acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # More aggressive early stopping
            if patience_counter >= max(5, self.config.PATIENCE // 2):
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Additional overfitting check - stop if gap is too large
            if epoch > 10 and train_acc - val_acc > 15:  # 15% gap threshold
                print(f"🛑 Stopping due to large train-val accuracy gap: {train_acc-val_acc:.1f}%")
                break
            
            # Save checkpoint every 5 epochs (reduced frequency)
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.config.SAVED_MODELS_DIR / f"resnet_checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Clear caches at end of epoch
            self.config.clear_gpu_cache()
            gc.collect()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        self.writer.close()
        return best_val_acc

if __name__ == "__main__":
    trainer = ResNetTrainer()
    trainer.train()