import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer
import nltk
from nltk.tree import Tree
import collections

class ConstituencyTreeParser:
    """选区树解析器 - 对应论文4.1.1节"""
    def __init__(self):
        self.parser = nltk.CoreNLPParser(url='http://localhost:9000')
    
    def parse_sentence(self, text):
        """将临床记录解析为选区树"""
        try:
            # 使用NLTK进行句法分析
            sentences = nltk.sent_tokenize(text)
            trees = []
            for sent in sentences:
                # 预处理：转换为小写，去重等
                sent = sent.lower().strip()
                if len(sent) > 1:
                    tree = next(self.parser.raw_parse(sent))
                    trees.append(tree)
            return trees
        except:
            # 备用解析方法
            return self._fallback_parse(text)
    
    def serialize_tree(self, tree):
        """将选区树序列化为二进制树 - 对应论文4.1.1节"""
        positions = []
        self._get_tree_positions(tree, positions, 0)
        return self._span_serialization(positions)
    
    def _get_tree_positions(self, tree, positions, start):
        """获取树节点位置"""
        if isinstance(tree, str):
            return start + 1
        else:
            end = start
            for child in tree:
                end = self._get_tree_positions(child, positions, end)
            positions.append((start, end))
            return end
    
    def _span_serialization(self, positions):
        """跨度序列化方法"""
        max_pos = max(end for _, end in positions) if positions else 0
        d_array = [0] * (2 * max_pos - 1)
        
        for left, right in positions:
            d_array[right - 1] = left
        return d_array

class BioBERTEncoder(nn.Module):
    """BioBERT编码器 - 对应论文4.1.2节和4.3节"""
    def __init__(self, model_name='dmis-lab/biobert-v1.1', hidden_size=768):
        super().__init__()
        self.biobert = BertModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def forward(self, texts):
        """对文本进行编码"""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded_outputs = []
        for text in texts:
            # 添加特殊标记
            tokens = self.tokenizer(text, return_tensors='pt', 
                                   padding=True, truncation=True, 
                                   max_length=512)
            
            with torch.no_grad():
                outputs = self.biobert(**tokens)
            
            # 取最后一层的隐藏状态
            hidden_states = outputs.last_hidden_state
            encoded_outputs.append(hidden_states)
        
        return torch.stack(encoded_outputs)

class TreeLSTMNode(nn.Module):
    """Tree-LSTM节点 - 对应论文4.2节公式"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # 输入门权重
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        
        # 遗忘门权重（每个子节点一个）
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(10)])  # 假设最多10个子节点
        
        # 输出门权重
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        
        # 更新单元权重
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
        
        # 偏置项
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        self.b_u = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, input_vector, child_states):
        """
        Tree-LSTM前向传播
        input_vector: 当前节点的输入向量 [batch_size, input_size]
        child_states: 子节点的(hidden_state, cell_state)元组列表
        """
        batch_size = input_vector.size(0)
        
        # 初始化子节点状态汇总
        if len(child_states) == 0:
            h_sum = torch.zeros(batch_size, self.hidden_size).to(input_vector.device)
            c_sum = torch.zeros(batch_size, self.hidden_size).to(input_vector.device)
        else:
            h_children = [h for h, c in child_states]
            c_children = [c for h, c in child_states]
            
            h_sum = sum(h_children)
            c_sum = sum(c_children)
        
        # 输入门
        i_j = torch.sigmoid(self.W_i(input_vector) + self.U_i(h_sum) + self.b_i)
        
        # 遗忘门（每个子节点独立）
        f_jk = []
        for k, (h_k, c_k) in enumerate(child_states):
            if k < len(self.U_f):
                f_k = torch.sigmoid(self.W_f(input_vector) + self.U_f[k](h_k) + self.b_f)
                f_jk.append(f_k)
        
        # 输出门
        o_j = torch.sigmoid(self.W_o(input_vector) + self.U_o(h_sum) + self.b_o)
        
        # 更新单元
        u_j = torch.tanh(self.W_u(input_vector) + self.U_u(h_sum) + self.b_u)
        
        # 记忆单元
        if len(child_states) == 0:
            c_j = i_j * u_j
        else:
            forget_sum = sum(f_k * c_k for f_k, (h_k, c_k) in zip(f_jk, child_states))
            c_j = i_j * u_j + forget_sum
        
        # 隐藏状态
        h_j = o_j * torch.tanh(c_j)
        
        return h_j, c_j

class TreeLSTM(nn.Module):
    """完整的Tree-LSTM模型"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.tree_lstm_node = TreeLSTMNode(input_size, hidden_size)
    
    def forward(self, tree_structure, input_vectors):
        """
        tree_structure: 序列化的选区树
        input_vectors: 对应的BioBERT向量
        """
        # 自底向上递归计算
        node_states = {}
        
        for node_id in reversed(range(len(tree_structure))):
            if tree_structure[node_id] == 0:  # 叶子节点
                input_vec = input_vectors[node_id]
                child_states = []
            else:
                input_vec = input_vectors[node_id]
                # 获取子节点状态
                child_indices = [i for i, parent in enumerate(tree_structure) 
                               if parent == node_id and i in node_states]
                child_states = [node_states[i] for i in child_indices]
            
            h_j, c_j = self.tree_lstm_node(input_vec, child_states)
            node_states[node_id] = (h_j, c_j)
        
        # 返回根节点的隐藏状态作为整个树的表示
        root_state = node_states[0][0] if 0 in node_states else input_vectors[0]
        return root_state

class TRICModel(nn.Module):
    """TRIC主模型 - 对应论文第4节整体架构"""
    def __init__(self, num_icd_codes, biobert_hidden_size=768, 
                 tree_lstm_hidden_size=256, dropout_rate=0.2):
        super().__init__()
        
        # BioBERT编码器
        self.biobert_encoder = BioBERTEncoder()
        
        # Tree-LSTM模块
        self.tree_lstm = TreeLSTM(biobert_hidden_size, tree_lstm_hidden_size)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(tree_lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_icd_codes)
        )
        
        # 选区树解析器
        self.tree_parser = ConstituencyTreeParser()
        
        # ICD代码编码（预计算）
        self.icd_embeddings = nn.Embedding(num_icd_codes, biobert_hidden_size)
        
        self.num_icd_codes = num_icd_codes
        self.dropout_rate = dropout_rate
    
    def forward(self, clinical_texts, icd_descriptions=None):
        """
        前向传播
        clinical_texts: 临床记录文本列表
        icd_descriptions: ICD代码描述（可选）
        """
        batch_size = len(clinical_texts)
        
        # 存储批量结果
        clinical_embeddings = []
        
        for text in clinical_texts:
            # 1. 构建选区树
            trees = self.tree_parser.parse_sentence(text)
            
            if not trees:
                # 如果解析失败，使用简单平均池化
                biobert_output = self.biobert_encoder(text)
                text_embedding = biobert_output.mean(dim=1)  # 平均池化
                clinical_embeddings.append(text_embedding)
                continue
            
            # 处理每个句子的选区树
            sentence_embeddings = []
            for tree in trees:
                # 序列化选区树
                serialized_tree = self.tree_parser.serialize_tree(tree)
                
                # 获取叶子节点的BioBERT编码
                leaf_texts = tree.leaves()
                if leaf_texts:
                    biobert_vectors = self.biobert_encoder(' '.join(leaf_texts))
                    leaf_embeddings = biobert_vectors.mean(dim=1)  # 句子级表示
                    
                    # Tree-LSTM处理
                    tree_embedding = self.tree_lstm(serialized_tree, leaf_embeddings)
                    sentence_embeddings.append(tree_embedding)
            
            if sentence_embeddings:
                # 平均所有句子的表示
                clinical_embedding = torch.stack(sentence_embeddings).mean(dim=0)
            else:
                biobert_output = self.biobert_encoder(text)
                clinical_embedding = biobert_output.mean(dim=1)
            
            clinical_embeddings.append(clinical_embedding)
        
        # 堆叠批量结果
        if clinical_embeddings:
            clinical_embeddings = torch.cat(clinical_embeddings, dim=0)
        else:
            # 备用方案：直接使用BioBERT编码
            biobert_output = self.biobert_encoder(clinical_texts)
            clinical_embeddings = biobert_output.mean(dim=1)
        
        # 分类输出
        logits = self.classifier(clinical_embeddings)
        
        return {
            'y_logit': logits,
            'clinical_embeddings': clinical_embeddings
        }
    
    def calculate_similarity(self, clinical_embeddings, icd_embeddings):
        """计算临床记录与ICD代码的相似度 - 对应论文4.4节"""
        # 线性变换对齐维度
        clinical_transformed = torch.matmul(clinical_embeddings, self.similarity_weight) + self.similarity_bias
        similarity_scores = F.cosine_similarity(clinical_transformed, icd_embeddings, dim=-1)
        return similarity_scores

class TRICClassifier:
    """TRIC分类器封装类，兼容BaseModel接口"""
    def __init__(self, num_classes, device=torch.device('cuda:0')):
        self.model = TRICModel(num_icd_codes=num_classes)
        self.device = device
        self.model.to(device)
        
        # 兼容BaseModel的模块列表
        self.moduleList = nn.ModuleList([self.model])
        self.crition = nn.BCEWithLogitsLoss()  # 多标签分类损失
    
    def calculate_y_logit(self, input_dict):
        """计算输出logits"""
        clinical_texts = input_dict['texts']
        return self.model(clinical_texts)
    
    def to_train_mode(self):
        """训练模式"""
        self.model.train()
    
    def to_eval_mode(self):
        """评估模式"""
        self.model.eval()
    
    def parameters(self):
        """返回模型参数"""
        return self.model.parameters()

# 辅助函数和类
class ClinicalDataProcessor:
    """临床数据处理类"""
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.biobert_tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 清理和标准化文本
        text = text.lower().strip()
        # 这里可以添加更多的文本清理逻辑
        return text
    
    def encode_batch(self, texts):
        """批量编码文本"""
        encoded = self.biobert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encoded

def test_model():
    """测试模型"""
    # 创建模型实例
    num_classes = 50  # ICD代码数量
    model = TRICClassifier(num_classes)
    
    # 测试数据
    test_texts = [
        "Male, diagnosed as pancreatic head cancer 8 months ago",
        "Patient with diabetes and hypertension"
    ]
    
    input_dict = {'texts': test_texts}
    
    # 前向传播测试
    model.to_eval_mode()
    with torch.no_grad():
        output = model.calculate_y_logit(input_dict)
        print("Output shape:", output['y_logit'].shape)
        print("Model test passed!")

if __name__ == "__main__":
    # 下载NLTK数据（首次运行需要）
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    test_model()
