import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Cấu hình các cột đặc trưng (Feature)
FEATURES = ['CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2'] 

def load_folder(folder_path):
    """
    Đọc tất cả file csv trong folder và chỉ lấy các cột FEATURES.
    Không cần inverse, không cần quan tâm scaler cũ.
    """
    search_path = os.path.join(folder_path, "*.csv")
    all_files = glob.glob(search_path)
    
    if len(all_files) == 0:
        print(f"CẢNH BÁO: Không tìm thấy file .csv nào trong '{folder_path}'")
        return pd.DataFrame()

    df_list = []
    
    print(f"Đang đọc {len(all_files)} files từ: {os.path.basename(folder_path)}")

    for filename in all_files:
        try:
            # 1. Đọc file
            df = pd.read_csv(filename)
            
            if df.empty: continue

            # 2. Kiểm tra cột
            # Chỉ lấy đúng các cột features cần thiết
            missing_cols = [col for col in FEATURES if col not in df.columns]
            
            if missing_cols:
                # Nếu thiếu cột quan trọng thì bỏ qua file này hoặc báo lỗi
                print(f"File {os.path.basename(filename)} thiếu cột: {missing_cols} -> Bỏ qua.")
                continue

            # 3. Lấy dữ liệu
            df_subset = df[FEATURES]
            df_list.append(df_subset)
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {os.path.basename(filename)}: {e}")

    if not df_list:
        return pd.DataFrame()

    # 4. Gộp tất cả lại
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    return merged_df

def load_train_val_test(root_folder):
    """
    Hàm chính để load dữ liệu cho 3 tập.
    Không cần tham số scaler_path nữa vì ta không dùng lại scaler cũ.
    """
    print(f"Đang đọc dữ liệu từ: {root_folder}")
    
    # Load từng folder
    print("--- Xử lý tập Train ---")
    df_train = load_folder(os.path.join(root_folder, "train"))
    
    print("--- Xử lý tập Validation ---")
    df_val = load_folder(os.path.join(root_folder, "validation")) 
    
    print("--- Xử lý tập Test ---")
    df_test = load_folder(os.path.join(root_folder, "test"))
    
    return df_train, df_val, df_test

def prepare_data(df_train, df_val, df_test): 
    """
    Input: Dữ liệu đã Standard Scaled.
    Output: Dữ liệu đã lọc ngoại lai và MinMax Scaled [0, 1].
    """
    std_scaler = joblib.load('./normalized_data/global_scaler.pkl')

    # 1. Copy dữ liệu train để xử lý
    df_train_clean = df_train.copy()
    
    # --- BƯỚC 1: LỌC NGOẠI LAI TRÊN TẬP TRAIN (Dựa trên giá trị đã Standardized) ---
    if not df_train_clean.empty:
        # Tính IQR
        Q1 = df_train_clean.quantile(0.25)
        Q3 = df_train_clean.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Lọc bỏ dòng ngoại lai
        condition = ~((df_train_clean < lower_bound) | (df_train_clean > upper_bound)).any(axis=1)
        
        n_before = len(df_train_clean)
        df_train_clean = df_train_clean[condition]
        
        print(f"Đã loại bỏ {n_before - len(df_train_clean)} dòng ngoại lai (dựa trên phân phối Standardized).")

    # --- BƯỚC 2: MIN-MAX SCALING ---
    print(" Đang thực hiện MinMaxScaler [0, 1]...")
    minmax_scaler = MinMaxScaler()
    
    # Fit trên tập train sạch
    X_train = minmax_scaler.fit_transform(df_train_clean)
    
    # Transform Val/Test (nếu có dữ liệu)
    X_val = minmax_scaler.transform(df_val) if not df_val.empty else None
    X_test = minmax_scaler.transform(df_test) if not df_test.empty else None
    X_train_raw = inverse_double_step(X_train, minmax_scaler, std_scaler, FEATURES) 
    X_val_raw = inverse_double_step(X_val, minmax_scaler, std_scaler, FEATURES) 
    X_test_raw = inverse_double_step(X_test, minmax_scaler, std_scaler, FEATURES) 
    joblib.dump(minmax_scaler, 'minmax_scaler.pkl')

    return X_train_raw, X_val_raw, X_test_raw, minmax_scaler

def inverse_double_step(data_minmax, mm_scaler, std_scaler, feature_names=FEATURES):
    """
    Hàm helper để thực hiện 2 bước inverse:
    MinMax [0,1] -> Standard [Z-score] -> Raw [Gốc]
    Đặc biệt: Xử lý vụ lệch cột (5 cột vs 14 cột).
    """
    # 1. Inverse MinMax (Dễ, vì mm_scaler vừa fit trên đúng 5 cột này)
    data_std = mm_scaler.inverse_transform(data_minmax)
    
    # 2. Inverse Standard (Khó, vì std_scaler cũ đòi 14 cột)
    if std_scaler is None:
        return data_std # Nếu không có scaler cũ thì trả về dạng Z-score
    
    # Lấy danh sách feature mà std_scaler cũ mong đợi
    # (Nếu scaler cũ không lưu feature_names_in_, bạn phải tự liệt kê list 14 cột cũ)
    try:
        expected_cols = std_scaler.feature_names_in_
        n_expected = len(expected_cols)
    except:
        # Fallback nếu sklearn phiên bản cũ
        n_expected = std_scaler.mean_.shape[0]
        expected_cols = [f"Col_{i}" for i in range(n_expected)]

    # Tạo ma trận giả full số 0 với kích thước (N_samples, 14)
    n_samples = data_std.shape[0]
    dummy_matrix = np.zeros((n_samples, n_expected))
    
    # Điền giá trị của 5 cột hiện tại vào đúng vị trí tương ứng trong ma trận giả
    # Cách đơn giản: Nếu bạn biết tên cột khớp nhau
    mapped_indices = []
    current_col_indices = []
    
    for i, col in enumerate(feature_names):
        # Tìm xem cột hiện tại (ví dụ 'CO') nằm ở vị trí nào trong scaler cũ
        for j, exp_col in enumerate(expected_cols):
            if col == exp_col:
                dummy_matrix[:, j] = data_std[:, i]
                mapped_indices.append(j)
                current_col_indices.append(i)
                break
    
    # Nếu không tìm thấy tên cột khớp (do tên khác nhau), code sẽ trả về Z-score
    if not mapped_indices:
        print("Cảnh báo: Tên cột không khớp với Scaler cũ. Không thể Inverse về Raw.")
        return data_std

    # Inverse Transform trên ma trận giả (14 cột)
    raw_dummy = std_scaler.inverse_transform(dummy_matrix)
    
    # Trích xuất lại đúng 5 cột chúng ta cần
    data_raw = np.zeros_like(data_std)
    for k, original_idx in enumerate(current_col_indices):
        target_idx = mapped_indices[k]
        data_raw[:, original_idx] = raw_dummy[:, target_idx]
        
    return data_raw