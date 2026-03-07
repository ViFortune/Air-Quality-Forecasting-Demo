import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def get_counts_from_paths(data_dirs):
    max_s, max_r = 0, 0
    all_files = []
    for d in data_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*_processed.csv")))
        
    for f in all_files:
        try:
            df = pd.read_csv(f, nrows=1)
            # Chỉ cần check Station và Region
            if 'station_id_encoded' in df.columns:
                max_s = max(max_s, int(df['station_id_encoded'].iloc[0]))
            if 'region_encoded' in df.columns:
                max_r = max(max_r, int(df['region_encoded'].iloc[0]))
        except Exception as e:
            continue
    return max_s + 1, max_r + 1

class AQIDualEmbeddingDataset(Dataset):
    def __init__(self, data_dir, sequence_length=24, target_col='VN_AQI'):
        self.sequence_length = sequence_length
        self.samples = []
        
        file_paths = glob.glob(os.path.join(data_dir, "*_processed.csv"))
        if not file_paths:
            print(f"Warning: Không tìm thấy data tại {data_dir}")

        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                
                # Lấy 2 chỉ số encoded
                s_idx = int(df['station_id_encoded'].iloc[0])
                r_idx = int(df['region_encoded'].iloc[0])
                
                # Bỏ các cột ID và Province ra khỏi input sequence
                exclude_cols = ['Date', 'station_id', 'region', 'province', 
                                'station_id_encoded', 'region_encoded', 'province_encoded', 
                                target_col]
                
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                
                features = df[feature_cols].values.astype(np.float32)
                targets = df[target_col].values.astype(np.float32)
                
                num_records = len(df)
                if num_records > sequence_length:
                    for i in range(num_records - sequence_length):
                        seq_data = features[i : i + sequence_length]
                        target_val = targets[i + sequence_length]
                        
                        self.samples.append({
                            'sequence': seq_data,
                            'station_idx': s_idx,
                            'region_idx': r_idx,
                            'target': target_val
                        })
            except Exception as e:
                continue
        
        if len(self.samples) > 0:
            self.input_dim = self.samples[0]['sequence'].shape[1]
        else:
            self.input_dim = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['sequence']),
            torch.tensor(sample['station_idx'], dtype=torch.long),
            torch.tensor(sample['region_idx'], dtype=torch.long),
            torch.tensor(sample['target'])
        )

def get_dataloaders(config):
    num_stations, num_regions = get_counts_from_paths(
        [config['train_dir'], config['val_dir'], config['test_dir']]
    )
    print(f"Metadata: {num_stations} stations, {num_regions} regions.")

    train_ds = AQIDualEmbeddingDataset(config['train_dir'], config['sequence_length'])
    val_ds = AQIDualEmbeddingDataset(config['val_dir'], config['sequence_length'])
    test_ds = AQIDualEmbeddingDataset(config['test_dir'], config['sequence_length'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    info = {
        'num_stations': num_stations,
        'num_regions': num_regions,
        'input_dim': train_ds.input_dim
    }
    return train_loader, val_loader, test_loader, info