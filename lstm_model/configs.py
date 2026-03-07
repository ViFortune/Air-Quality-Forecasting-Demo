import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    info_path = './ai_engine/data/origin/dataset_info.json'
    train_dir = './ai_engine/data/train/'
    val_dir   = './ai_engine/data/validation/'
    test_dir  = './ai_engine/data/test/'
    
    save_model_path = './ai_engine/weights/best_aqi_model_dual.pth'

    scheduler_factor = 0.5
    scheduler_patience = 6
    min_lr = 1e-7
    warmup_epochs = 3
    warmup_start_lr = 1e-5
    grad_clip = 1.0
    loss_type = 'huber'
    
    # Embedding dims
    sequence_length = 24
    embedding_dim_station = 10
    embedding_dim_region = 4
    embedding_dim_province = 5  
    
    # Model structure
    hidden_dim = 64
    num_layers = 2
    bidirectional = True
    
    dropout = 0.2               
    batch_size = 32
    learning_rate = 0.001
    
    epochs = 100
    patience = 25
    
    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = ModelConfig()