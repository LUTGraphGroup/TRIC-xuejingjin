import torch
import json
import logging
import os
import random
import numpy as np
from datetime import datetime

def setup_device(use_gpu=True):
    """设置训练设备 - CPU或GPU"""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def setup_logging(log_dir='logs'):
    """设置日志记录"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_config(config, filepath):
    """保存训练配置到JSON文件"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """从JSON文件加载配置"""
    with open(filepath, 'r') as f:
        return json.load(f)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """估算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

class EarlyStopper:
    """早停机制实现"""
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # min mode
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def print_model_summary(model, input_shape):
    """打印模型摘要信息"""
    print(f"{'Model Summary':=^50}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size(model):.2f} MB")
    print(f"Input shape: {input_shape}")
    print("=" * 50)
