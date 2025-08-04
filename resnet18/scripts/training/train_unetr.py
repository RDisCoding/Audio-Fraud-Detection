import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import time
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.config import Config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class AudioTransformer(nn.Module):
    def __init__(self, patch_dim, d_model=512, nhead=8, num_layers=6, num_classes=2):
        super(AudioTransformer, self).__init__()
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        batch_size, seq_len, patch_dim = x.shape
        
        # Embed patches
        x = self.patch_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)  # (batch_size, seq_len+1, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len+1, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len+1, d_model)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use class token for classification
        cls_output = x[:, 0]  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(cls_output)
        
        return output

class UNETRTrainer:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load dataset stats to get patch dimension
        stats_path = self.config.PREPROCESSED_DIR / "unetr" / "dataset_stats.pkl"
        with open(stats_path, 'rb') as f:
            self.dataset_stats = pickle.load(f)
        
        # Initialize model
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=self.config.LEARNING_RATE, 
                                   weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS
        )
        
        # Setup logging
        self.writer = SummaryWriter(self.config.LOGS_DIR / 'unetr')
        
    def _build_model(self):
        """Build UNETR model for audio fraud detection"""
        patch_dim = self.dataset_stats['patch_dimension']
        
        model = AudioTransformer(
            patch_dim=patch_dim,
            d_model=256,  # Reduced for memory efficiency
            nhead=8,
            num_layers=4,  # Reduced for memory efficiency
            num_classes=2
        )
        
        return model.to(self.device)
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        
        # Load training data
        train_path = self.config.PREPROCESSED_DIR / "unetr" / "training_data.pkl"
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        
        # Load validation data
        val_path = self.config.PREPROCESSED_DIR / "unetr" / "validation_data.pkl"
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        # Convert to PyTorch tensors
        train_X = torch.FloatTensor(train_data['data'])
        train_y = torch.LongTensor(train_data['labels'])
        val_X = torch.FloatTensor(val_data['data'])
        val_y = torch.LongTensor(val_data['labels'])
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        # Use smaller batch size for transformer to manage memory
        batch_size = max(1, self.config.BATCH_SIZE // 4)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with gradient checkpointing for memory efficiency
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self):
        """Full training loop"""
        print("Starting UNETR training...")
        
        # Load data
        train_loader, val_loader = self.load_data()
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                model_path = self.config.SAVED_MODELS_DIR / "unetr_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': self.config,
                    'dataset_stats': self.dataset_stats
                }, model_path)
                
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                print(f"Model saved to {model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.config.SAVED_MODELS_DIR / f"unetr_checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'dataset_stats': self.dataset_stats
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        self.writer.close()
        return best_val_acc

if __name__ == "__main__":
    trainer = UNETRTrainer()
    trainer.train()