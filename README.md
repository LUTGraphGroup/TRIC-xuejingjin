# TRIC-xuejingjin
ICD code mapping model based on clinical text tree structure

1. Project Overview

This project proposes an ICD code mapping model leveraging the tree structure of clinical text (e.g., hierarchical medical terminology). It integrates data preprocessing, tree-structured feature extraction, and deep learning to automatically map unstructured clinical notes to ICD codes.  

2. Usage Workflow

Follow these steps to execute the model:  

Step 1: Preprocess Raw Clinical Data

Use data_processor.py to clean and format raw datasets (e.g., MIMIC-III).  
from data_processor import MIMICProcessor

# Initialize with raw data path
mimic = MIMICProcessor(data_path="path/to/mimic-iii")
# Generate preprocessed CSV
mimic.get_basic_data(out_path="data.csv")
  

Step 2: Prepare Data Class for Modeling

Use data_processor.py to create a data utility class (handles word embeddings, dataset splitting).  
from data_processor import DataUtils

data_utils = DataUtils(
    data_path="data.csv",
    mimic_path="path/to/mimic-iii",
    stopwords_path="stopwords.txt",
    valid_size=0.2,   # Validation set ratio
    test_size=0.2,    # Test set ratio
    min_count=10      # Minimum word frequency
)
  

Step 3: Compute ICD Vectors (Leverage Tree Structure)

Use tree_parser.py to extract tree-structured features from ICD descriptions, then generate ICD vectors.  
from tree_parser import ICDDescriptor

icd_desc = ICDDescriptor(
    data_class=data_utils,
    mimic_path="path/to/mimic-iii"
)
if data_utils.class_num == 50:
    icd_desc = icd_desc[data_utils.icd_index]
  

Step 4: Train the Tree-Structured Model

Use training.py to build and train the model (e.g., Tree-LSTM, GNN).  
from training import Trainer

trainer = Trainer(
    data_utils=data_utils,
    icd_desc=icd_desc,
    embed_dim=128,   # Word embedding dimension
    hidden_dim=256,  # Model hidden layer dimension
    epochs=50        # Training epochs
)
trainer.train()
  

Step 5: Predict ICD Codes for New Texts

Use main.py to load the trained model and make predictions.  
from main import predict_icd

text = "Patient has fever and cough."  # Input clinical text
predicted_codes = predict_icd(text)
print(f"Predicted ICD codes: {predicted_codes}")
  

3. File Responsibilities

File Core Function
config.py Centralize hyperparameters (learning rate, batch size, tree depth, etc.).
data_processor.py Clean raw data, tokenize text, and split datasets (train/val/test).
tree_parser.py Parse clinical text into tree structures (map terms to medical ontologies).
utils.py Utility functions (file I/O, embedding loading, logging).
metrics.py Define evaluation metrics (accuracy, F1-score for multi-label ICD).
training.py Implement training loop (loss function, optimizer, early stopping).
main.py Orchestrate end-to-end workflow (preprocess → train → predict).
  
