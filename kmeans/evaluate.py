from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import load_data as data
import clustering
import visualize
import joblib


def calculate_silhouette(X, labels):
    """
    Tính Silhouette Score.
    Range: [-1, 1]. Càng gần 1 càng tốt (các cụm tách biệt rõ).
    """
    if X is None or len(X) == 0:
        return 0.0
    # Silhouette yêu cầu ít nhất 2 nhãn khác nhau và số lượng mẫu > số nhãn
    try:
        score = silhouette_score(X, labels)
        return score
    except Exception as e:
        print(f"Không thể tính Silhouette Score: {e}")
        return 0.0

def evaluate_model(model, X, dataset_name="Dataset"):
    """
    Hàm tổng hợp: Dự báo nhãn và tính toán các chỉ số đánh giá.
    Sử dụng cho cả Validation và Test set.
    """
    if X is None or len(X) == 0:
        print(f"{dataset_name}: Không có dữ liệu để đánh giá.")
        return None

    # 1. Dự báo nhãn (Predict)
    labels = model.predict(X)

    # 2. Tính toán các chỉ số (Metrics)
    sil_score = calculate_silhouette(X, labels)
    
    # (Optional) Tính thêm Davies-Bouldin nếu cần (càng thấp càng tốt)
    db_score = davies_bouldin_score(X, labels)

    # 3. In ra console (để debug)
    print(f">> Silhouette Score trên tập {dataset_name}: {sil_score:.4f}")
    print(f">> Davies-Bouldin Score trên tập {dataset_name}: {db_score:.4f}")

    # 4. Trả về kết quả (dạng dict để Web UI dễ lấy hiển thị)
    return {
        "dataset": dataset_name,
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "labels": labels
    }

def main():
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    root_folder = "./data" 

    # Kiểm tra đường dẫn có tồn tại không
    if not os.path.exists(root_folder):
        print(f"LỖI: Đường dẫn '{root_folder}' không tồn tại! Hãy kiểm tra lại.")
        return

    # --- BƯỚC 1: LOAD DỮ LIỆU ---
    print("\n--- 1. Loading Data ---")
    df_train, df_val, df_test = data.load_train_val_test(root_folder)
    
    if df_train.empty:
        print("Lỗi: Tập Train rỗng. Kiểm tra lại thư mục 'train'.")
        return

    print(f"Số lượng mẫu Train: {len(df_train)}")
    print(f"Số lượng mẫu Val:   {len(df_val)}")
    print(f"Số lượng mẫu Test:  {len(df_test)}")

    # --- BƯỚC 2: TIỀN XỬ LÝ (Fit trên Train) ---
    print("\n--- 2. Preprocessing ---")
    X_train, X_val, X_test, scaler = data.prepare_data(df_train, df_val, df_test)
    # --- BƯỚC 3: KHÁM PHÁ (Elbow) ---
    print("\n--- 3. Running Elbow Method (Mining Phase) ---")
    # Chỉ chạy Elbow trên tập Train để tìm cấu trúc
    wcss = clustering.run_elbow_method(X_train, k_range=range(1, 11))
    visualize.plot_elbow(range(1, 11), wcss)
    
    # --- BƯỚC 4: HUẤN LUYỆN (Deployment Phase) ---
    print("\n--- 4. Training K-Means ---")
    kmeans_model, labels, centers = clustering.train_kmeans(X_train, k=6)
    
    # Vẽ heatmap tâm cụm để so sánh với bảng quy chuẩn
    visualize.plot_centroids_heatmap(kmeans_model, scaler, data.FEATURES)


    # --- BƯỚC 5: ĐÁNH GIÁ (Evaluation) ---
    print("\n--- 5. Evaluating on Val and Test Set ---")
    
    # Đánh giá tập Val
    if X_val is not None:
        evaluate_model(kmeans_model, X_val, dataset_name="Validation")

    # Đánh giá tập Test
    if X_test is not None:
        test_results = evaluate_model(kmeans_model, X_test, dataset_name="Test")
        
        # Lấy labels từ kết quả trả về để vẽ hình
        test_labels = test_results["labels"]
        
        # Vẽ PCA cho tập test (nếu muốn)
        visualize.visualize_clusters_pca(X_test, test_labels, centers)
        visualize.visualize_cluster_features(X_test, test_labels, data.FEATURES)

if __name__ == "__main__":
    main()