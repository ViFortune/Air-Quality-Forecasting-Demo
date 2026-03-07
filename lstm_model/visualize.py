import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves(train_losses, val_losses, save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Training loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation loss', color='orange', linewidth=2)
    
    plt.title('Learning curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ loss tại: {save_path}")

def plot_prediction_comparison(targets, predictions, station_name='Station', save_dir='./results', limit=200):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    if hasattr(targets, 'cpu'): targets = targets.cpu().numpy()
    if hasattr(predictions, 'cpu'): predictions = predictions.cpu().numpy()
    
    t = targets[:limit]
    p = predictions[:limit]
    
    plt.plot(t, label='Thực tế (Ground Truth)', color='green', alpha=0.7)
    plt.plot(p, label='Dự báo (Prediction)', color='red', linestyle='dashed')
    
    plt.title(f'So sánh Thực tế vs Dự báo - {station_name}', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('VN_AQI', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, f'prediction_compare_{station_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Đã lưu biểu đồ dự báo tại: {save_path}")