import torch

# 实验配置 - 对应论文Table 3的超参数设置
class ExperimentConfig:
    """实验配置类，包含所有超参数设置"""
    
    def __init__(self):
        # 数据配置
        self.data_path = "data/mimic-iii"
        self.max_seq_length = 512
        self.batch_size = 32
        self.train_ratio = 0.8
        
        # 模型架构配置 - 对应论文4.1-4.4节
        self.biobert_model_name = "dmis-lab/biobert-v1.1"
        self.biobert_hidden_size = 768
        self.tree_lstm_hidden_size = 256
        self.num_icd_codes = 8907  # MIMIC-III的ICD代码数量
        
        # 训练配置 - 对应论文5.4节
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        self.dropout_rate = 0.2
        self.num_epochs = 100
        self.early_stop_patience = 10
        self.lr_patience = 5
        
        # 评估配置 - 对应论文5.3节
        self.threshold = 0.2
        self.metrics = ["MiF", "MaF", "MiAUC", "MaAUC", "P@8"]
        
        # 设备配置
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed = 42
        
        # 路径配置
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"
        self.result_dir = "results"
    
    def to_dict(self):
        """将配置转换为字典"""
        return {
            'data_path': self.data_path,
            'max_seq_length': self.max_seq_length,
            'batch_size': self.batch_size,
            'train_ratio': self.train_ratio,
            'biobert_model_name': self.biobert_model_name,
            'biobert_hidden_size': self.biobert_hidden_size,
            'tree_lstm_hidden_size': self.tree_lstm_hidden_size,
            'num_icd_codes': self.num_icd_codes,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.num_epochs,
            'early_stop_patience': self.early_stop_patience,
            'lr_patience': self.lr_patience,
            'threshold': self.threshold,
            'metrics': self.metrics,
            'device': str(self.device),
            'seed': self.seed
        }
    
    def from_dict(self, config_dict):
        """从字典加载配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

# 默认配置实例
default_config = ExperimentConfig()

# MIMIC-III 50样本数据集配置
mimic_50_config = ExperimentConfig()
mimic_50_config.num_icd_codes = 50
mimic_50_config.batch_size = 16
mimic_50_config.num_epochs = 50

# 消融实验配置
ablation_config = ExperimentConfig()
ablation_config.tree_lstm_hidden_size = 128
ablation_config.dropout_rate = 0.3
