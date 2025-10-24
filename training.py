import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import numpy as np
from metrics import calculate_metrics
import warnings
warnings.filterwarnings('ignore')

class TRICTrainer:
    """TRIC模型训练器 - 实现论文5.4节的训练流程"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 优化器和损失函数
        self.optimizer = Adam(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=config['lr_patience'], 
            factor=0.5
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 训练状态
        self.best_val_score = 0.0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            texts = batch['text']
            targets = batch['icd_vector'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(texts)
            loss = self.criterion(outputs['y_logit'], targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % self.config['log_interval'] == 0:
                current_time = time.time() - start_time
                print(f'Epoch: {epoch} [{batch_idx * len(texts)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)] '
                      f'Loss: {loss.item():.6f} Time: {current_time:.2f}s')
        
        avg_loss = total_loss / batch_count
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                texts = batch['text']
                targets = batch['icd_vector'].to(self.device)
                
                outputs = self.model(texts)
                loss = self.criterion(outputs['y_logit'], targets)
                
                total_loss += loss.item()
                
                # 收集预测结果
                preds = torch.sigmoid(outputs['y_logit'])
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算指标
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = calculate_metrics(all_targets, all_preds, threshold=0.2)
        avg_loss = total_loss / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        self.val_scores.append(metrics['MiF'])  # 使用Micro F1作为主要指标
        
        print(f'Validation - Epoch: {epoch}, Loss: {avg_loss:.6f}, '
              f'MiF: {metrics["MiF"]:.4f}, MaF: {metrics["MaF"]:.4f}, '
              f'MiAUC: {metrics["MiAUC"]:.4f}, P@8: {metrics["P@8"]:.4f}')
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        print("Starting training...")
        print(f"Training config: {self.config}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练阶段
            train_loss = self.train_epoch(epoch)
            
            # 验证阶段
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step(val_metrics['MiF'])
            
            # 早停检查
            if val_metrics['MiF'] > self.best_val_score:
                self.best_val_score = val_metrics['MiF']
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, True)
                print(f'New best model saved with MiF: {self.best_val_score:.4f}')
            else:
                self.epochs_no_improve += 1
                self.save_checkpoint(epoch, False)
            
            # 检查早停条件
            if self.epochs_no_improve >= self.config['early_stop_patience']:
                print(f'Early stopping at epoch {epoch}')
                break
        
        print(f"Training completed. Best MiF: {self.best_val_score:.4f}")
    
    def save_checkpoint(self, epoch, is_best):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        
        if is_best:
            torch.save(checkpoint, 'best_model.pth')
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_score = checkpoint['best_val_score']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

def setup_training_config():
    """训练配置 - 对应论文Table 3超参数设置"""
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'early_stop_patience': 10,
        'lr_patience': 5,
        'log_interval': 10,
        'biobert_hidden_size': 768,
        'tree_lstm_hidden_size': 256,
        'dropout_rate': 0.2
    }
    return config
