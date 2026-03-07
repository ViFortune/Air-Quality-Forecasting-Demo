import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib # Dùng để lưu và tải mô hình
import seaborn as sns
import xgboost

'''
    In this file, we handle the task which predict day by day and plot them into 1 plot with the ground truth, then the second task is that bases on the previous prediction to predict the next, next day to forecast.... 
'''

def calculate_aqi_vn_from_subindices(pm25_aqi, pm10_aqi, co_aqi, so2_aqi):
    """
    Tính AQI theo cách hiển thị thực tế trên cem.gov.vn
    Input: AQI con của từng chất (đã được trang web tính sẵn)
    Output: AQI tổng + chất gây ô nhiễm chính
    """
    aqi_values = {
        'PM2.5': pm25_aqi,
        'PM10': pm10_aqi,
        'CO': co_aqi,
        'SO2': so2_aqi
    }
    
    aqi_total = max(aqi_values.values())
    
    dominant = max(aqi_values, key=aqi_values.get)
    
    return {
        'VN_AQI': int(aqi_total),
        'dominant_pollutant': dominant,
        'sub_aqi': aqi_values
    }


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

def predict_next_7_days(ckpt_dir, data_dir):
    csv_obj = pd.read_csv(data_dir+'processed_HN_KDT_KK.csv', encoding='utf-8')
    latest_day = pd.to_datetime(csv_obj['Date'][0])

    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']
    model_params = {
        'CO': f'{ckpt_dir}xgboost_model_CO.pkl',
        'PM-10': f'{ckpt_dir}xgboost_model_PM-10.pkl',
        'PM-2-5': f'{ckpt_dir}xgboost_model_PM-2-5.pkl',
        'SO2': f'{ckpt_dir}xgboost_model_SO2.pkl',
    }
    real_time_path = f'{data_dir}real_time_v1.pt'
    X_train, Y_train_dict = load_data(real_time_path)
    print(len(X_train))
    model_CO = joblib.load(model_params['CO'])
    model_PM10 = joblib.load(model_params['PM-10'])
    model_PM25 = joblib.load(model_params['PM-2-5'])
    model_SO2 = joblib.load(model_params['SO2'])

    # First we should do with the PM-10 for the first demo
    # But the test set is the whole test sample from all station, so we have to modified the specific station
    dict_of_list = {
        'CO': [],
        'PM-10': [],
        'PM-2-5': [],
        'SO2': []
    }

    for i, name in enumerate(targets):        
        for example_idx in range(len(X_train)):
            if example_idx == 0:
                dict_of_list[name].extend([X_train[example_idx][j] for j in range(i, 31+i, 5)])

            dict_of_list[name].append(Y_train_dict[name][example_idx])

    X_lagged = None
    for day in range(1, 8):
        curr_date = latest_day + pd.Timedelta(days=day)
        latest_day = curr_date
        weekday = curr_date.weekday()
        month = curr_date.month

        if day == 1:
            X_lagged = np.array([X_train[len(X_train)-1]]) # get the latest 7 days

        CO_pred = model_CO.predict(X_lagged)[0]
        PM10_pred = model_PM10.predict(X_lagged)[0]
        PM25_pred = model_PM25.predict(X_lagged)[0]
        SO2_pred = model_SO2.predict(X_lagged)[0]

        dict_of_list['CO'].append(CO_pred)
        dict_of_list['PM-10'].append(PM10_pred)
        dict_of_list['PM-2-5'].append(PM25_pred)
        dict_of_list['SO2'].append(SO2_pred)

        # Tính AQI
        result = calculate_aqi_vn_from_subindices(PM25_pred, PM10_pred, SO2_pred, CO_pred)
        aqi = result['VN_AQI']

        predict_components = np.array([CO_pred, PM10_pred, PM25_pred, SO2_pred, aqi], dtype=np.float32)

        weekday_feature = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32) #mon,tu,wed,thu,fri,sat,sun
        region_feature = np.array([1, 0, 0], dtype=np.float32) # north,middle,south
        seasons = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32) # spring,summer,autumn,winter,dry,rain 

        weekday_feature[weekday] = 1
        seasons[0] = 1 if month in [1,2,3] else 0
        seasons[1] = 1 if month in [4,5,6] else 0
        seasons[2] = 1 if month in [7,8,9] else 0 
        seasons[3] = 1 if month in [10,11,12] else 0

        lagged = np.concatenate((predict_components, X_lagged[0][0:30]))
        rest_features = np.concatenate((weekday_feature, region_feature, seasons))
                
        X_lagged = np.array([np.concatenate((lagged, rest_features), axis=0, dtype=np.float32)])

    return dict_of_list

def visualize(plots_dir, data_dir, dict_of_list: dict[list]):
    # Thiết lập style đẹp
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Đọc dữ liệu
    # csv_path = data_dir + 'processed_HN_NguyenVanCu_28560877461938780203765592307.csv'
    csv_path = data_dir + 'processed_HN_KDT_KK.csv'
    csv_obj = pd.read_csv(csv_path, encoding='utf-8')
    csv_obj['Date'] = pd.to_datetime(csv_obj['Date'])

    # Lấy ngày mới nhất
    latest_day = csv_obj['Date'].iloc[0]

    # Tạo danh sách ngày: 29 ngày trước + 1 latest + 7 ngày sau = 37 ngày
    day_list = []
    for i in range(1, 20):  # base on the n_records of the station
        day_list.append(latest_day - pd.Timedelta(days=i))
    day_list.append(latest_day)  # bao gồm ngày latest
    for i in range(1, 8):
        day_list.append(latest_day + pd.Timedelta(days=i))

    # Sắp xếp theo thời gian
    day_list = sorted(day_list)
    date_str_list = [d.strftime('%Y-%m-%d') for d in day_list]

    # Tạo DataFrame
    plot_data = pd.DataFrame(dict_of_list, index=date_str_list)
    plot_data.index.name = 'Date'

    # Xác định chỉ số của latest_day và các ngày dự đoán
    latest_idx = date_str_list.index(latest_day.strftime('%Y-%m-%d'))
    predict_start_idx = latest_idx + 1  # từ ngày tiếp theo
    predict_end_idx = len(date_str_list)  # đến hết

    # Màu sắc
    base_color = '#1f77b4'  # Màu lam cho tất cả các chất (thực tế + latest)
    predict_color = '#ff7f0e'  # Màu cam cho 7 ngày dự đoán
    latest_color = 'red'  # Màu đỏ cho ngày latest

    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']

    # Tạo và lưu 4 biểu đồ riêng biệt (mỗi pollutant 1 file)
    for idx, pollutant in enumerate(targets):
        # Tạo figure riêng cho từng pollutant
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))  # Kích thước phù hợp cho 1 biểu đồ
        values = plot_data[pollutant].values
        bars = ax.bar(date_str_list, values, color=base_color, edgecolor='black', linewidth=0.6, alpha=0.9)

        # Tô màu cho 7 ngày dự đoán
        for i in range(predict_start_idx, predict_end_idx):
            bars[i].set_color(predict_color)
            bars[i].set_edgecolor('darkorange')
            bars[i].set_alpha(0.85)

        # Highlight ngày latest
        bars[latest_idx].set_color(latest_color)
        bars[latest_idx].set_edgecolor('darkred')
        bars[latest_idx].set_linewidth(2.5)

        # Xoay nhãn ngày
        ax.tick_params(axis='x', rotation=45)

        # Tiêu đề
        ax.set_title(f'AQI of {pollutant} through 37 days\n'
                     f'(Blue: Real value | Red: Latest day | Orange: Predict next 7 days)',
                     fontsize=14, fontweight='bold', pad=20)

        # Nhãn trục
        ax.set_ylabel('AQI', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)

        # Lưới
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Thêm giá trị trên cột (kiểm tra NaN)
        valid_values = values[~np.isnan(values)]
        max_val = valid_values.max() if len(valid_values) > 0 else 1
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                offset = max_val * 0.01
                ax.text(bar.get_x() + bar.get_width()/2., val + offset,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Điều chỉnh layout
        plt.tight_layout()

        # Lưu PDF riêng cho từng pollutant
        pdf_path = plots_dir + f'aqi_{pollutant}.png'
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Saved plot for {pollutant} at: {pdf_path}")

        # # Hiển thị (tùy chọn, sẽ hiện từng cái một)
        # plt.show()
        # plt.close(fig)  # Đóng figure để tránh chồng chéo

    print("All 4 separate charts saved successfully!")
    

if __name__ == '__main__':
    dict_of_list = predict_next_7_days('./model/XGBoost/ckpts/', './data/')
    visualize('./static/plots/', './data/', dict_of_list)