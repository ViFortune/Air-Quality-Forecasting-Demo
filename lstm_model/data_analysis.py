import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_data_distribution(train_loader, val_loader, test_loader):
    """
    Phân tích phân phối dữ liệu của các datasets
    """
    
    def extract_targets(loader, max_samples=None):
        targets = []
        count = 0
        for _, _, _, target in loader:
            targets.extend(target.numpy())
            count += len(target)
            if max_samples and count >= max_samples:
                break
        return np.array(targets)
    
    print("Extracting targets from datasets...")
    train_targets = extract_targets(train_loader, max_samples=5000)
    val_targets = extract_targets(val_loader)
    test_targets = extract_targets(test_loader)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_targets)}")
    print(f"  Val:   {len(val_targets)}")
    print(f"  Test:  {len(test_targets)}")
    
    # Statistical comparison
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    datasets = {
        'Train': train_targets,
        'Val': val_targets,
        'Test': test_targets
    }
    
    print(f"\n{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12}")
    print("-"*60)
    
    for name, data in datasets.items():
        print(f"\n{name} Statistics:")
        print(f"  Mean:        {data.mean():.4f}")
        print(f"  Std:         {data.std():.4f}")
        print(f"  Min:         {data.min():.4f}")
        print(f"  Max:         {data.max():.4f}")
        print(f"  Median:      {np.median(data):.4f}")
        print(f"  Q25:         {np.percentile(data, 25):.4f}")
        print(f"  Q75:         {np.percentile(data, 75):.4f}")
        print(f"  Skewness:    {stats.skew(data):.4f}")
        print(f"  Kurtosis:    {stats.kurtosis(data):.4f}")
        
        # Count high values
        high_count = (data > 1.0).sum()
        high_pct = high_count / len(data) * 100
        print(f"  High (>1.0): {high_count} ({high_pct:.1f}%)")
    
    # Statistical tests
    print("\n" + "="*60)
    print("DISTRIBUTION TESTS")
    print("="*60)
    
    # Kolmogorov-Smirnov test
    ks_val_test = stats.ks_2samp(val_targets, test_targets)
    ks_train_val = stats.ks_2samp(train_targets, val_targets)
    ks_train_test = stats.ks_2samp(train_targets, test_targets)
    
    print(f"\nKolmogorov-Smirnov Test (p-value):")
    print(f"  Val vs Test:   {ks_val_test.pvalue:.4f} {'✓ Same' if ks_val_test.pvalue > 0.05 else '✗ Different'}")
    print(f"  Train vs Val:  {ks_train_val.pvalue:.4f} {'✓ Same' if ks_train_val.pvalue > 0.05 else '✗ Different'}")
    print(f"  Train vs Test: {ks_train_test.pvalue:.4f} {'✓ Same' if ks_train_test.pvalue > 0.05 else '✗ Different'}")
    
    # T-test for means
    t_val_test = stats.ttest_ind(val_targets, test_targets)
    print(f"\nT-test (means equal, p-value):")
    print(f"  Val vs Test:   {t_val_test.pvalue:.4f} {'✓ Same' if t_val_test.pvalue > 0.05 else '✗ Different'}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histograms
    ax = axes[0, 0]
    bins = np.linspace(
        min(train_targets.min(), val_targets.min(), test_targets.min()),
        max(train_targets.max(), val_targets.max(), test_targets.max()),
        50
    )
    ax.hist(train_targets, bins=bins, alpha=0.5, label='Train', density=True)
    ax.hist(val_targets, bins=bins, alpha=0.5, label='Val', density=True)
    ax.hist(test_targets, bins=bins, alpha=0.5, label='Test', density=True)
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plots
    ax = axes[0, 1]
    ax.boxplot([train_targets, val_targets, test_targets], 
               labels=['Train', 'Val', 'Test'])
    ax.set_ylabel('Target Value')
    ax.set_title('Box Plot Comparison')
    ax.grid(True, alpha=0.3)
    
    # 3. CDF comparison
    ax = axes[1, 0]
    for name, data in datasets.items():
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=name, alpha=0.7)
    ax.set_xlabel('Target Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q plot (Val vs Test)
    ax = axes[1, 1]
    quantiles = np.linspace(0, 100, 100)
    val_quantiles = np.percentile(val_targets, quantiles)
    test_quantiles = np.percentile(test_targets, quantiles)
    ax.scatter(val_quantiles, test_quantiles, alpha=0.6)
    ax.plot([val_quantiles.min(), val_quantiles.max()], 
            [val_quantiles.min(), val_quantiles.max()], 
            'r--', label='Perfect match')
    ax.set_xlabel('Val Quantiles')
    ax.set_ylabel('Test Quantiles')
    ax.set_title('Q-Q Plot: Val vs Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved: data_distribution_analysis.png")
    plt.show()
    
    # Diagnosis
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS")
    print("="*60)
    
    if ks_val_test.pvalue < 0.05:
        print("⚠️  Val and Test have DIFFERENT distributions!")
        print("   → This explains the gap between Val loss and Test MSE")
        
        mean_diff = abs(val_targets.mean() - test_targets.mean())
        std_diff = abs(val_targets.std() - test_targets.std())
        
        if mean_diff > 0.1:
            print(f"   → Mean difference: {mean_diff:.4f} (significant)")
        if std_diff > 0.1:
            print(f"   → Std difference: {std_diff:.4f} (significant)")
            
        val_high_pct = (val_targets > 1.0).sum() / len(val_targets) * 100
        test_high_pct = (test_targets > 1.0).sum() / len(test_targets) * 100
        
        if abs(val_high_pct - test_high_pct) > 5:
            print(f"   → Val has {val_high_pct:.1f}% high values")
            print(f"   → Test has {test_high_pct:.1f}% high values")
            print("   → WeightedMSELoss penalizes high values more in Val!")
    else:
        print("✓ Val and Test have similar distributions")
        print("  → The gap is likely due to the loss function difference")
    
    print("="*60)
    
    return train_targets, val_targets, test_targets


# Usage in your main script:
# train_targets, val_targets, test_targets = analyze_data_distribution(
#     train_loader, val_loader, test_loader
# )