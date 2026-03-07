import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
def plot_elbow(k_range, wcss):
    """Vẽ biểu đồ khuỷu tay."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
    plt.title('Phương pháp Elbow tìm K tối ưu')
    plt.xlabel('Số lượng cụm (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.show()

def plot_centroids_heatmap(model, scaler, feature_names):
    """Vẽ Heatmap giá trị trung bình của các cụm."""
    # Lấy tâm cụm (đang ở dạng scaled)
    centroids_original = model.cluster_centers_
    
    # Tạo DataFrame để vẽ
    df_centroids = pd.DataFrame(centroids_original, columns=feature_names)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_centroids, annot=True, fmt=".1f", cmap="YlOrBr")
    plt.title("Giá trị trung bình các đặc trưng của 6 Cụm (Centroids)")
    plt.ylabel("Cụm (Cluster)")
    plt.xlabel("Đặc trưng")
    plt.show()

def visualize_cluster_features(X_scaled, labels, feature_names):
    # Tạo DataFrame từ dữ liệu scaled
    df_vis = pd.DataFrame(X_scaled, columns=feature_names)
    df_vis['Cluster'] = labels
    
    # Chuyển đổi dữ liệu sang dạng dài (long format) để vẽ seaborn
    df_melt = df_vis.melt(id_vars='Cluster', var_name='Feature', value_name='Normalized Value')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melt, x='Feature', y='Normalized Value', hue='Cluster', palette='viridis')
    
    plt.title('Phân bố các chỉ số ô nhiễm trong từng cụm')
    plt.ylabel('Giá trị chuẩn hóa (0-1)')
    plt.xlabel('Loại khí / Bụi')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def visualize_clusters_pca(X_scaled, labels, centers=None):
    # Giảm chiều dữ liệu từ 5D -> 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Tạo DataFrame để dễ vẽ
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ các điểm dữ liệu
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.6, s=50)
    
    # Vẽ tâm cụm (nếu có)
    if centers is not None:
        centers_pca = pca.transform(centers)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
    
    plt.title('Phân cụm dữ liệu ô nhiễm không khí (PCA 2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()