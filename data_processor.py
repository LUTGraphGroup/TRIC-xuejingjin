import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import nltk
from nltk.tree import Tree
import re

class MIMICIIIDataset(Dataset):
    """MIMIC-III数据集处理类 - 对应论文5.1节数据集描述"""
    
    def __init__(self, data_path, max_length=512, is_train=True):
        self.data_path = data_path
        self.max_length = max_length
        self.is_train = is_train
        self.clinical_notes = []
        self.icd_codes = []
        self.icd_descriptions = {}
        
        # 加载ICD代码描述
        self._load_icd_descriptions()
        # 加载临床数据
        self._load_data()
        
    def _load_icd_descriptions(self):
        """加载ICD代码描述 - 对应论文4.3节"""
        # 这里应该从ICD官方描述文件加载
        # 示例数据
        self.icd_descriptions = {
            '001.0': 'Cholera due to vibrio cholerae',
            '001.1': 'Cholera due to vibrio cholerae el tor',
            # ... 其他ICD代码描述
        }
    
    def _load_data(self):
        """加载MIMIC-III数据 - 对应论文Table 2"""
        try:
            # 从CSV文件加载数据
            if self.is_train:
                data_df = pd.read_csv(f'{self.data_path}/train.csv')
            else:
                data_df = pd.read_csv(f'{self.data_path}/test.csv')
            
            for _, row in data_df.iterrows():
                self.clinical_notes.append(row['text'])
                self.icd_codes.append(row['icd_codes'].split(';'))
                
        except FileNotFoundError:
            print("Warning: Data file not found, using sample data")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据用于测试"""
        sample_texts = [
            "Male, diagnosed as pancreatic head cancer 8 months ago, recently died of hemorrhagic shock due to biliary obstruction and upper gastrointestinal bleeding.",
            "Female patient with diabetes mellitus and hypertension, admitted for routine checkup."
        ]
        sample_icds = [['157.0', '456.0'], ['250.00', '401.9']]
        
        self.clinical_notes = sample_texts
        self.icd_codes = sample_icds
    
    def __len__(self):
        return len(self.clinical_notes)
    
    def __getitem__(self, idx):
        text = self.clinical_notes[idx]
        icd_list = self.icd_codes[idx]
        
        # 创建多标签向量
        icd_vector = torch.zeros(len(self.icd_descriptions))
        for icd in icd_list:
            if icd in self.icd_descriptions:
                icd_idx = list(self.icd_descriptions.keys()).index(icd)
                icd_vector[icd_idx] = 1
        
        return {
            'text': text,
            'icd_vector': icd_vector,
            'icd_list': icd_list
        }

class ClinicalTextPreprocessor:
    """临床文本预处理器 - 对应论文4.1节预处理"""
    
    def __init__(self):
        self.tree_parser = ConstituencyTreeParser()
        
    def preprocess_text(self, text):
        """文本预处理：清洗、分词、构建选区树"""
        # 清洗文本
        text = self._clean_text(text)
        
        # 构建选区树
        trees = self.tree_parser.parse_sentence(text)
        
        # 序列化选区树
        serialized_trees = []
        for tree in trees:
            serialized = self.tree_parser.serialize_tree(tree)
            serialized_trees.append(serialized)
        
        return {
            'original_text': text,
            'trees': trees,
            'serialized_trees': serialized_trees,
            'leaf_texts': [tree.leaves() for tree in trees] if trees else [text.split()]
        }
    
    def _clean_text(self, text):
        """清理临床文本"""
        # 转换为小写
        text = text.lower()
        # 移除特殊字符但保留医学术语
        text = re.sub(r'[^\w\s.,;:-]', '', text)
        # 标准化空格
        text = ' '.join(text.split())
        return text

def create_data_loaders(data_path, batch_size=32, train_ratio=0.8):
    """创建训练和验证数据加载器"""
    full_dataset = MIMICIIIDataset(data_path, is_train=True)
    
    # 分割数据集
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, full_dataset.icd_descriptions
