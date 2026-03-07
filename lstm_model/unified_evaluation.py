import torch
import torch.nn as nn
import numpy as np

def comprehensive_evaluation(model, data_loader, device, criterion=None, dataset_name="Test"):
    """
    Đánh giá toàn diện với nhiều metrics
    """
    model.eval()
    
    # Initialize metrics
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    
    predictions = []
    actuals = []
    weighted_losses = []
    
    with torch.no_grad():
        for seqs, s_idxs, r_idxs, targets in data_loader:
            seqs = seqs.to(device)
            s_idxs = s_idxs.to(device)
            r_idxs = r_idxs.to(device)
            targets = targets.to(device)
            
            preds = model(seqs, s_idxs, r_idxs)
            
            # Store predictions
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            
            # Calculate weighted loss if criterion provided
            if criterion is not None:
                weighted_loss = criterion(preds, targets)
                weighted_losses.append(weighted_loss.item())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate standard metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # Calculate additional metrics
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - actuals.mean()) ** 2))
    
    # Metrics for high values (> threshold)
    threshold = 1.0
    high_mask = actuals > threshold
    if high_mask.sum() > 0:
        high_mse = np.mean((predictions[high_mask] - actuals[high_mask]) ** 2)
        high_mae = np.mean(np.abs(predictions[high_mask] - actuals[high_mask]))
        high_count = high_mask.sum()
    else:
        high_mse = high_mae = high_count = 0
    
    # Metrics for low values (<= threshold)
    low_mask = ~high_mask
    if low_mask.sum() > 0:
        low_mse = np.mean((predictions[low_mask] - actuals[low_mask]) ** 2)
        low_mae = np.mean(np.abs(predictions[low_mask] - actuals[low_mask]))
        low_count = low_mask.sum()
    else:
        low_mse = low_mae = low_count = 0
    
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SET EVALUATION")
    print(f"{'='*60}")
    print(f"Overall Metrics:")
    print(f"  MSE:                 {mse:.4f}")
    print(f"  RMSE:                {rmse:.4f}")
    print(f"  MAE:                 {mae:.4f}")
    print(f"  MAPE:                {mape:.2f}%")
    print(f"  R²:                  {r2:.4f}")
    
    if criterion is not None and weighted_losses:
        avg_weighted_loss = np.mean(weighted_losses)
        print(f"  Weighted Loss:       {avg_weighted_loss:.4f}")
    
    print(f"\nHigh Values (>{threshold}):")
    print(f"  Count:               {high_count} ({high_count/len(actuals)*100:.1f}%)")
    if high_count > 0:
        print(f"  MSE:                 {high_mse:.4f}")
        print(f"  MAE:                 {high_mae:.4f}")
    
    print(f"\nLow Values (<={threshold}):")
    print(f"  Count:               {low_count} ({low_count/len(actuals)*100:.1f}%)")
    if low_count > 0:
        print(f"  MSE:                 {low_mse:.4f}")
        print(f"  MAE:                 {low_mae:.4f}")
    
    print(f"{'='*60}\n")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'high_mse': high_mse,
        'high_mae': high_mae,
        'low_mse': low_mse,
        'low_mae': low_mae,
        'weighted_loss': np.mean(weighted_losses) if weighted_losses else None
    }


def compare_datasets(model, train_loader, val_loader, test_loader, device, criterion):
    """
    So sánh metrics giữa các datasets để tìm vấn đề
    """
    print("\n" + "="*60)
    print("COMPARING ALL DATASETS")
    print("="*60)
    
    # Sample from train set (để tránh quá lâu)
    train_subset = []
    for i, batch in enumerate(train_loader):
        train_subset.append(batch)
        if i >= 10:  # Chỉ lấy 10 batches
            break
    
    class SubsetLoader:
        def __init__(self, batches):
            self.batches = batches
        def __iter__(self):
            return iter(self.batches)
    
    train_sample_loader = SubsetLoader(train_subset)
    
    # Evaluate each dataset
    print("\n1. Training Set (sample):")
    train_metrics = comprehensive_evaluation(model, train_sample_loader, device, criterion, "Train Sample")
    
    print("\n2. Validation Set:")
    val_metrics = comprehensive_evaluation(model, val_loader, device, criterion, "Validation")
    
    print("\n3. Test Set:")
    test_metrics = comprehensive_evaluation(model, test_loader, device, criterion, "Test")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
    print("-"*60)
    print(f"{'MSE':<20} {train_metrics['mse']:<12.4f} {val_metrics['mse']:<12.4f} {test_metrics['mse']:<12.4f}")
    print(f"{'RMSE':<20} {train_metrics['rmse']:<12.4f} {val_metrics['rmse']:<12.4f} {test_metrics['rmse']:<12.4f}")
    print(f"{'MAE':<20} {train_metrics['mae']:<12.4f} {val_metrics['mae']:<12.4f} {test_metrics['mae']:<12.4f}")
    
    if criterion is not None:
        print(f"{'Weighted Loss':<20} {train_metrics['weighted_loss']:<12.4f} {val_metrics['weighted_loss']:<12.4f} {test_metrics['weighted_loss']:<12.4f}")
    
    print("="*60)
    
    # Diagnose issues
    print("\n🔍 DIAGNOSIS:")
    
    # Check for data distribution mismatch
    if abs(val_metrics['mse'] - test_metrics['mse']) > 0.2:
        print("⚠️  WARNING: Large gap between Val MSE and Test MSE")
        print("   → Check data distribution and preprocessing")
    
    # Check if weighted loss explains the gap
    if criterion is not None:
        val_weighted = val_metrics['weighted_loss']
        test_standard = test_metrics['mse']
        
        if val_weighted > test_standard * 2:
            print("⚠️  WARNING: Weighted loss much higher than standard MSE")
            print("   → The high_val_weight penalty is causing the gap")
            print(f"   → Val uses WeightedMSE={val_weighted:.4f}")
            print(f"   → Test uses standard MSE={test_standard:.4f}")
    
    # Check overfitting
    if train_metrics['mse'] < val_metrics['mse'] * 0.7:
        print("⚠️  WARNING: Possible overfitting")
        print("   → Train MSE much lower than Val MSE")
    
    return train_metrics, val_metrics, test_metrics