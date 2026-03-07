import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
import joblib # Dùng để lưu và tải mô hình


# 1. Tải và chuẩn bị dữ liệu (Giả định)
def load_data(data_path):
    # Giả định file lưu dữ liệu dưới dạng dict
    ds = torch.load(data_path, map_location='cpu', weights_only=False)
    
    # Chuyển đổi từ tensor/numpy về numpy array
    X = np.array(ds["X"])
    Y = {
        'CO': np.array(ds["CO"]),
        'PM-10': np.array(ds["PM-10"]),
        'PM-2-5': np.array(ds["PM-2-5"]),
        'SO2': np.array(ds["SO2"])
    }
    return X, Y

# 2. Định nghĩa hàm huấn luyện
def train_xgboost_models(X_train, Y_train_dict, targets, params):
    trained_models = {}
    
    for target in targets:
        print(f"--- Bắt đầu huấn luyện mô hình cho: {target} ---")
        
        # Lấy dữ liệu mục tiêu tương ứng
        Y_train = Y_train_dict[target]
        
        # Khởi tạo mô hình XGBoost
        model = xgb.XGBRegressor(
            objective='reg:squarederror', # Mục tiêu hồi quy
            n_estimators=1000, # Số lượng cây (Có thể tuning)
            learning_rate=0.05, # Tốc độ học (Có thể tuning)
            n_jobs=-1, # Sử dụng tất cả các nhân CPU
            random_state=42,
            **params.get(target, {}) # Thêm các tham số riêng nếu có
        )
        
        # Huấn luyện mô hình
        model.fit(X_train, Y_train)
        
        # Lưu mô hình đã huấn luyện
        model_filename = 'ckpts/'+f'xgboost_model_{target}.pkl'
        joblib.dump(model, model_filename)
        print(f"Hoàn thành và lưu mô hình tại: {model_filename}")
        
        trained_models[target] = model
        
    return trained_models

if __name__ == '__main__':
    # --- Chạy Quy trình Training ---
    train_path = 'data/train.pt' # Thay thế đường dẫn thực tế
    test_path = 'data/test.pt'
    X_train, Y_train_dict = load_data(train_path)
    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']
    params = {} # Bạn có thể thêm các tham số tuning riêng tại đây

    trained_models = train_xgboost_models(X_train, Y_train_dict, targets, params)

    # --- Plot prediction after training ---
    if True == False:
        ds = torch.load(train_path, weights_only=False)
        example_index = 7
        X_CO_lagged = np.array([ds["X"][example_index]])
        CO_lagged = np.array([ds["X"][example_index][i] for i in range(0, 31, 5)])
        CO_target = np.array(ds["CO"][example_index])
        CO_mean = np.mean(CO_lagged)

        # Tải mô hình
        model = joblib.load('xgboost_model_CO.pkl')

        # 1. Dự đoán
        CO_pred = model.predict(X_CO_lagged)

        plt.figure(figsize=(7,3))
        plt.plot(range(1, 8, 1), CO_lagged, 'go', color='blue', markersize=3)
        plt.plot([8], CO_target, color='red', marker='^', markersize=3)
        plt.plot([8], CO_mean, color='black', marker='x', markersize=3)
        plt.plot([8], CO_pred, color='green', marker='o', markersize=3)

        plt.plot()