import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib # Dùng để lưu và tải mô hình
from train import load_data


# 3. Định nghĩa hàm dự đoán và đánh giá
def evaluate_xgboost_models(test_path, targets):
    
    # Tải dữ liệu kiểm tra
    X_test, Y_test_dict = load_data(test_path)
    
    metrics = {}
    all_predictions = {}
    
    print("\n--- Bắt đầu đánh giá trên tập Test ---")
    
    for target in targets:
        model_filename = f'xgboost_model_{target}.pkl'
        
        try:
            # Tải mô hình
            model = joblib.load(model_filename)
            
            # 1. Dự đoán
            Y_pred = model.predict(X_test)
            Y_true = Y_test_dict[target]
            
            # 2. Tính toán Metrics
            rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))
            mae = mean_absolute_error(Y_true, Y_pred)
            r2 = r2_score(Y_true, Y_pred)
            
            # 3. Lưu kết quả
            metrics[target] = {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            }
            all_predictions[target] = Y_pred
            
            print(f"Đã đánh giá {target}: RMSE={rmse:.4f}, MAE={mae:.4f}")
            
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy mô hình cho {target} tại {model_filename}. Vui lòng train trước.")
            return None, None

    return metrics, all_predictions

if __name__ == '__main__':
    # --- Chạy Quy trình Testing và Đánh giá ---
    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']
    test_path = 'path/to/your/test_data.pt' # Thay thế đường dẫn thực tế

    # Chú ý: Bạn cần chạy hàm train_xgboost_models() trước khi chạy hàm này
    metrics, predictions = evaluate_xgboost_models(test_path, targets)

    # --- Hiển thị kết quả (Sau khi chạy xong hàm evaluate) ---
    if metrics:
        print("\n=============================================")
        print("         KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG          ")
        print("=============================================")
        
        total_mae = 0
        
        for target, m in metrics.items():
            print(f"\n[Dự đoán: {target}]")
            print(f"  RMSE: {m['RMSE']:.4f}")
            print(f"  MAE:  {m['MAE']:.4f}")
            print(f"  R2:   {m['R2']:.4f}")
            total_mae += m['MAE']
            
        avg_mae = total_mae / len(targets)
        print("\n---------------------------------------------")
        print(f"Sai số tuyệt đối trung bình tổng quát (Avg MAE): {avg_mae:.4f} (Đây là chỉ số tổng quan)")
        print("=============================================")