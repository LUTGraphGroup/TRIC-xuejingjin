import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

class MetricCalculator:
    """评估指标计算器 - 精确实现论文5.3节的指标定义"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def micro_f1(self, y_true, y_pred, threshold=0.2):
        """计算Micro F1分数 - 对应论文MiF指标"""
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Micro平均：将所有类别的预测合并计算
        micro_precision = precision_score(y_true.flatten(), y_pred_binary.flatten(), 
                                        average='micro', zero_division=0)
        micro_recall = recall_score(y_true.flatten(), y_pred_binary.flatten(), 
                                  average='micro', zero_division=0)
        
        if micro_precision + micro_recall == 0:
            return 0.0
        
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        return micro_f1
    
    def macro_f1(self, y_true, y_pred, threshold=0.2):
        """计算Macro F1分数 - 对应论文MaF指标"""
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Macro平均：先计算每个类别的F1，再平均
        macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        return macro_f1
    
    def micro_auc(self, y_true, y_pred):
        """计算Micro AUC - 对应论文MiAUC指标"""
        try:
            # Micro AUC：将所有类别的预测合并计算
            micro_auc = roc_auc_score(y_true.flatten(), y_pred.flatten(), 
                                    average='micro')
            return micro_auc
        except ValueError:
            return 0.5  # 无法计算时返回随机值
    
    def macro_auc(self, y_true, y_pred):
        """计算Macro AUC - 对应论文MaAUC指标"""
        try:
            # Macro AUC：先计算每个类别的AUC，再平均
            macro_auc = roc_auc_score(y_true, y_pred, average='macro')
            return macro_auc
        except ValueError:
            return 0.5
    
    def precision_at_k(self, y_true, y_pred, k=8):
        """计算P@k指标 - 对应论文P@8指标"""
        precision_scores = []
        
        for i in range(len(y_true)):
            true_labels = y_true[i]
            pred_probs = y_pred[i]
            
            # 获取预测概率最高的k个标签
            top_k_indices = np.argsort(pred_probs)[-k:][::-1]
            
            # 计算在这些位置上的精度
            relevant = 0
            for idx in top_k_indices:
                if true_labels[idx] == 1:
                    relevant += 1
            
            precision_at_k = relevant / k if k > 0 else 0.0
            precision_scores.append(precision_at_k)
        
        return np.mean(precision_scores)
    
    def calculate_all_metrics(self, y_true, y_pred, threshold=0.2):
        """计算所有评估指标"""
        metrics = {
            'MiF': self.micro_f1(y_true, y_pred, threshold),
            'MaF': self.macro_f1(y_true, y_pred, threshold),
            'MiAUC': self.micro_auc(y_true, y_pred),
            'MaAUC': self.macro_auc(y_true, y_pred),
            'P@5': self.precision_at_k(y_true, y_pred, k=5),
            'P@8': self.precision_at_k(y_true, y_pred, k=8)
        }
        
        return metrics
    
    def print_metrics(self, metrics, prefix=""):
        """格式化打印指标结果"""
        print(f"{prefix}Metrics: "
              f"MiF={metrics['MiF']:.4f}, "
              f"MaF={metrics['MaF']:.4f}, "
              f"MiAUC={metrics['MiAUC']:.4f}, "
              f"MaAUC={metrics['MaAUC']:.4f}, "
              f"P@5={metrics['P@5']:.4f}, "
              f"P@8={metrics['P@8']:.4f}")

# 兼容性函数
def calculate_metrics(y_true, y_pred, threshold=0.2):
    """兼容旧代码的接口函数"""
    calculator = MetricCalculator(y_true.shape[1])
    return calculator.calculate_all_metrics(y_true, y_pred, threshold)

if __name__ == "__main__":
    # 测试指标计算
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3]])
    
    calculator = MetricCalculator(3)
    metrics =
calculator.calculate_all_metrics(y_true, y_pred)
calculator.print_metrics(metrics, "Test ")
