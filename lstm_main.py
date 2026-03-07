import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
import numpy as np
from tqdm import tqdm

from lstm_model.data_utils import get_dataloaders
from lstm_model.model import DualEmbeddingBiLSTM, WeightedMSELoss 
from lstm_model.configs import cfg
from lstm_model.visualize import plot_learning_curves, plot_prediction_comparison

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def warmup_lr_scheduler(epoch, warmup_epochs, warmup_start_lr, base_lr):
    if epoch < warmup_epochs:
        return warmup_start_lr + (base_lr - warmup_start_lr) * epoch / warmup_epochs
    return base_lr

def print_final_report(best_loss, best_epoch, total_epochs):
    print(f"\n{'Summary':^40}")
    print("="*40)
    print(f"Tổng số Epochs        : {total_epochs}")
    print(f"Epoch tốt nhất        : {best_epoch}")
    print(f"Loss tốt nhất         : {best_loss:.6f}")
    print("="*40 + "\n")

def save_results_to_pkl(train_hist, val_hist, best_epoch, best_loss, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_history.pkl")
    config_dict = cfg.__dict__ if hasattr(cfg, '__dict__') else {}
    data = {
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "config": {k: v for k, v in config_dict.items() if not k.startswith('__')}
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Đã xuất file kết quả tại: {save_path}")

def evaluate_test_set(model, test_loader, device):
    model.eval()
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    total_mse, total_mae = 0, 0
    predictions, actuals = [], []
    
    with torch.no_grad():
        for seqs, s_idxs, r_idxs, targets in test_loader:
            seqs, s_idxs, r_idxs, targets = seqs.to(device), s_idxs.to(device), r_idxs.to(device), targets.to(device)
            
            preds = model(seqs, s_idxs, r_idxs)
            
            total_mse += criterion_mse(preds, targets).item() * seqs.size(0)
            total_mae += criterion_mae(preds, targets).item() * seqs.size(0)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    num_samples = len(test_loader.dataset)
    final_mse = total_mse / num_samples
    final_mae = total_mae / num_samples
    final_rmse = np.sqrt(final_mse)
    
    print(f"\nFinal test metrics:")
    print(f"   MSE:     {final_mse:.4f}")
    print(f"   RMSE:    {final_rmse:.4f}")
    print(f"   MAE:     {final_mae:.4f}")
    print("-" * 45)
    return actuals, predictions

def run_training():
    seed_everything(42) 
    
    # Configs
    warmup_epochs = getattr(cfg, 'warmup_epochs', 3)
    warmup_start_lr = getattr(cfg, 'warmup_start_lr', 1e-5)
    grad_clip = getattr(cfg, 'grad_clip', 1.0)
    weight_decay = getattr(cfg, 'weight_decay', 1e-4) 
    
    print(f"Device: {cfg.device} | Batch: {cfg.batch_size} | Hidden: {cfg.hidden_dim}")

    config_dict = {
        'info_path': cfg.info_path, 'train_dir': cfg.train_dir,
        'val_dir': cfg.val_dir, 'test_dir': cfg.test_dir,
        'sequence_length': cfg.sequence_length, 'batch_size': cfg.batch_size
    }
    
    train_loader, val_loader, test_loader, data_info = get_dataloaders(config_dict)
    if data_info['input_dim'] == 0: return

    model = DualEmbeddingBiLSTM(
        config=cfg,
        num_stations=data_info['num_stations'],
        num_regions=data_info['num_regions'],
        input_dim=data_info['input_dim']
    ).to(cfg.device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(cfg.epochs):
        if epoch < warmup_epochs:
            lr = warmup_lr_scheduler(epoch, warmup_epochs, warmup_start_lr, cfg.learning_rate)
            for pg in optimizer.param_groups: pg['lr'] = lr
        
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.epochs}", leave=False)
        
        for seqs, s_idxs, r_idxs, targets in train_bar:
            seqs, s_idxs, r_idxs, targets = seqs.to(cfg.device), s_idxs.to(cfg.device), r_idxs.to(cfg.device), targets.to(cfg.device)
            
            optimizer.zero_grad()
            predictions = model(seqs, s_idxs, r_idxs)
            loss = criterion(predictions, targets)
            loss.backward()
            if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_train_loss = np.mean(train_losses)
        
        model.eval()
        val_loss_sum, val_mse_sum = 0, 0
        with torch.no_grad():
            for seqs, s_idxs, r_idxs, targets in val_loader:
                seqs, s_idxs, r_idxs, targets = seqs.to(cfg.device), s_idxs.to(cfg.device), r_idxs.to(cfg.device), targets.to(cfg.device)
                preds = model(seqs, s_idxs, r_idxs)
                val_loss_sum += criterion(preds, targets).item()
                val_mse_sum += torch.nn.functional.mse_loss(preds, targets).item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_mse = val_mse_sum / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch >= warmup_epochs: scheduler.step(avg_val_loss)
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        
        print(f"Ep {epoch+1:03d} | Train: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.4f} | LR: {current_lr:.6f}", end="")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_model_path)
            print(" | Saved Best")
        else:
            patience_counter += 1
            print(f" | Pat: {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at Epoch {epoch+1}")
                break
                
    print(f"Model saved: {cfg.save_model_path}")
    model.load_state_dict(torch.load(cfg.save_model_path))
    test_actuals, test_preds = evaluate_test_set(model, test_loader, cfg.device)
    
    return (model, train_loader, val_loader, test_loader, criterion,
            train_loss_history, val_loss_history, best_epoch, best_val_loss, 
            test_actuals, test_preds)
    
if __name__ == "__main__":
    (model, train_loader, val_loader, test_loader, criterion, 
     t_hist, v_hist, best_ep, best_loss, 
     test_act, test_pred) = run_training()
    
    if t_hist and len(t_hist) > 0:
        print_final_report(best_loss, best_ep, cfg.epochs)
        save_results_to_pkl(t_hist, v_hist, best_ep, best_loss)
        plot_learning_curves(t_hist, v_hist)
        if len(test_act) > 0:
            plot_prediction_comparison(
                torch.tensor(test_act), 
                torch.tensor(test_pred), 
                station_name="FINAL_TEST_RESULT_DUAL"
            )
    # from model.data_analysis import analyze_data_distribution
    # analyze_data_distribution(train_loader, val_loader, test_loader)
    
    # from model.unified_evaluation import compare_datasets
    # compare_datasets(model, train_loader, val_loader, test_loader, cfg.device, criterion)