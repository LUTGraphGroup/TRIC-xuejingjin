#!/usr/bin/env python3
"""
TRIC模型主训练脚本 - 对应论文第5节实验部分
用法: python main.py --config config.json --mode train
"""

import argparse
import torch
import os
import json
from data_processor import create_data_loaders
from ClassifierModel import TRICClassifier
from training import TRICTrainer, setup_training_config
from utils import setup_device, setup_logging, set_seed, save_config
from config import default_config, mimic_50_config
from metrics import MetricCalculator

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TRIC Model Training')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'predict'], help='运行模式')
    parser.add_argument('--data_path', type=str, default='data/mimic-iii', help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--dataset', type=str, default='full', choices=['full', '50'], help='数据集类型')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device == 'cuda')
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logging()
    logger.info("Starting TRIC Model Training")
    logger.info(f"Arguments: {args}")
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = default_config.from_dict(config_dict)
    else:
        config = mimic_50_config if args.dataset == '50' else default_config
    
    # 更新命令行参数
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.device = device
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    train_loader, val_loader, icd_descriptions = create_data_loaders(
        config.data_path, 
        config.batch_size, 
        config.train_ratio
    )
    
    # 创建模型
    logger.info("Creating TRIC model...")
    model = TRICClassifier(
        num_classes=len(icd_descriptions),
        device=config.device
    )
    
    # 打印模型信息
    from utils import print_model_summary
    print_model_summary(model.model, (config.batch_size, config.max_seq_length))
    
    if args.mode == 'train':
        # 训练配置
        training_config = setup_training_config()
        training_config.update(config.to_dict())
        
        # 创建训练器
        trainer = TRICTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config.device,
            config=training_config
        )
        
        # 开始训练
        logger.info("Starting training process...")
        trainer.train()
        
        # 保存最终配置
        save_config(config.to_dict(), os.path.join(config.checkpoint_dir, 'final_config.json'))
        
    elif args.mode == 'test':
        # 测试模式
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        
        # 运行测试
        metrics_calculator = MetricCalculator(len(icd_descriptions))
        all_preds = []
        all_targets = []
        
        model.to_eval_mode()
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                targets = batch['icd_vector'].to(config.device)
                
                outputs = model.calculate_y_logit({'texts': texts})
                preds = torch.sigmoid(outputs['y_logit'])
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = metrics_calculator.calculate_all_metrics(all_targets, all_preds)
        metrics_calculator.print_metrics(metrics, "Test ")
        
    elif args.mode == 'predict':
        # 预测模式
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        
        # 示例预测
        sample_texts = [
            "Male patient with chest pain and shortness of breath",
            "Female with diabetes and hypertension"
        ]
        
        model.to_eval_mode()
        with torch.no_grad():
            outputs = model.calculate_y_logit({'texts': sample_texts})
            predictions = torch.sigmoid(outputs['y_logit'])
            
            for i, text in enumerate(sample_texts):
                print(f"Text: {text}")
                print("Predicted ICD codes:")
                for j, prob in enumerate(predictions[i]):
                    if prob > 0.2:  # 阈值
                        icd_code = list(icd_descriptions.keys())[j]
                        desc = icd_descriptions[icd_code]
                        print(f"  {icd_code}: {desc} (prob: {prob:.3f})")
                print()
    
    logger.info("Program completed successfully")

if __name__ == "__main__":
    main()
