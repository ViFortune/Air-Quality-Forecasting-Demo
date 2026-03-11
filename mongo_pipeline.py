import os
import sys
import pandas as pd
import requests
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('.')

from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

from plot_prediction import calculate_aqi_vn_from_subindices

load_dotenv(dotenv_path=Path('.env'))

MONGO_URI = str(os.getenv("MONGO_URI"))
DB_NAME = str(os.getenv("DB_NAME"))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))
LAST_RUN_TIME = str(os.getenv("LAST_RUN_TIME"))

def get_mongo_client():
    from pymongo import MongoClient
    client = MongoClient(host=MONGO_URI)
    return client

def set_last_run_time() -> str:
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[LAST_RUN_TIME]

    run_time = datetime.datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%H:%M:%S -- %d/%m/%Y")
    # drop all previous records, we just want to keep the latest run time
    collection.delete_many({})
    collection.insert_one({
        "last_run_time": run_time
    })

    return run_time

def get_last_run_time() -> str:
    client = get_mongo_client()
    db = client[DB_NAME]
    collection = db[LAST_RUN_TIME]

    record = collection.find_one({
        "last_run_time": {"$exists": True}
    })

    return record["last_run_time"] if record else "Have no run yet!"

def crawl_data(date=30, type=0) -> pd.DataFrame:
    '''
        :param type: 0 if get average value by day, 1 to get exactly value by hours (max 26 value) 
    '''
    # define url
    url = f"https://envisoft.gov.vn/eip/default/call/json/get_aqi_data%3Fdate%3D{date}%26aqi_type%3D{type}"

    num_station = 1

    for i in range(num_station):

        station_id = 31390908889087377344742439468
        alias = "HN_KDT_KK" # Ha Noi, Khuat Duy Tien KK
        add_col = "CO,NO2,O3,PM-10,PM-2-5,SO2"
        station_name = "Ha Noi: Khuat Duy Tien KK"
        components = list(add_col.split(","))

        data = {
            "sEcho": "1",
            "iColumns": "9",
            "sColumns": ",,,,,,,,",
            "iDisplayStart": "0",
            "iDisplayLength": f"{date}",
            "mDataProp_0": "0",
            "sSearch_0": "",
            "bRegex_0": "false",
            "bSearchable_0": "true",
            "mDataProp_1": "1",
            "sSearch_1": "",
            "bRegex_1": "false",
            "bSearchable_1": "true",
            "mDataProp_2": "2",
            "sSearch_2": "",
            "bRegex_2": "false",
            "bSearchable_2": "true",
            "mDataProp_3": "3",
            "sSearch_3": "",
            "bRegex_3": "false",
            "bSearchable_3": "true",
            "mDataProp_4": "4",
            "sSearch_4": "",
            "bRegex_4": "false",
            "bSearchable_4": "true",
            "mDataProp_5": "5",
            "sSearch_5": "",
            "bRegex_5": "false",
            "bSearchable_5": "true",
            "mDataProp_6": "6",
            "sSearch_6": "",
            "bRegex_6": "false",
            "bSearchable_6": "true",
            "mDataProp_7": "7",
            "sSearch_7": "",
            "bRegex_7": "false",
            "bSearchable_7": "true",
            "mDataProp_8": "8",
            "sSearch_8": "",
            "bRegex_8": "false",
            "bSearchable_8": "true",
            "sSearch": "",
            "bRegex": "false",
            "station_id": f"{station_id}", 
            "from_date": "",
            "added_columns": add_col
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:148.0) Gecko/20100101 Firefox/148.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Connection": "keep-alive"
        }

        response = requests.post(url=url, data=data, headers=headers, timeout=30)

        if response.status_code == 200:

            try: 
                result = response.json()
                aaData = result["aaData"]  # Array of array
                cols = ["STT","Date","VN_AQI"] + components
                df = pd.DataFrame(data=aaData, columns=cols)
                print(df.head(n=30))

                print(f"{station_name} has been recorded with number of samples is {result['iTotalRecords']-1}\n")

                return df

            except ValueError:
                print("Responsed value is not JSON format, do not store")
                return None

        print(f"Error code: {response.status_code}!\n")
        print(response.text)
        return None

def drop_col(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.T.drop_duplicates().T
    df = df.drop(columns=["STT"], axis=1)
    df = df.drop(columns=['NO2'], axis=1) if 'NO2' in df.columns else df
    df = df.drop(columns=['O3'], axis=1) if 'O3' in df.columns else df
    
    for attr in df.columns.to_list():
        arr: np
        arr = df[attr].to_numpy()
        miss_rate = np.where(arr == '-')[0].shape[0]/df.shape[0]
        
        if miss_rate > 0.9:
            print(f"Remove column [{attr}] with [{miss_rate*100:.5}%] missing rate")
            df = df.drop(columns=[attr], axis=1)

    return df

def feature_engineering_and_preprocessing(df: pd.DataFrame, region: str) -> pd.DataFrame:
    '''
        This function is responsible for adding nefw columns to our csv files
        Adding some field to robust our 
        
        Adding region addtribute: north, middle, south. These take from the attribute in the file
    '''
    df = df
    df = df.drop_duplicates()
    df = df.replace('-', pd.NA)     # for interpolation progress
    
    # Transform date into DOW
    df.loc[:, 'Date'] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    month = df['Date'].dt.month
    # print(df['Date'].to_list())
    DOW = df['Date'].dt.day_of_week
    df.insert(1, 'Day_Of_Week', DOW)

    df.insert(df.shape[1], 'mon', df["Day_Of_Week"].isin([0]).astype(int))
    df.insert(df.shape[1], 'tu', df["Day_Of_Week"].isin([1]).astype(int))
    df.insert(df.shape[1], 'wed', df["Day_Of_Week"].isin([2]).astype(int))
    df.insert(df.shape[1], 'thu', df["Day_Of_Week"].isin([3]).astype(int))
    df.insert(df.shape[1], 'fri', df["Day_Of_Week"].isin([4]).astype(int))
    df.insert(df.shape[1], 'sat', df["Day_Of_Week"].isin([5]).astype(int))
    df.insert(df.shape[1], 'sun', df["Day_Of_Week"].isin([6]).astype(int))

    
    north_flag, middle_flag, south_flag = 0, 0, 0
    if region == 'north':
        north_flag = 1
    
    elif region == 'middle':
        middle_flag = 1
        
    else:
        south_flag = 1

    df.insert(df.shape[1], 'north', north_flag)
    df.insert(df.shape[1], 'middle', middle_flag)
    df.insert(df.shape[1], 'south', south_flag)
    
    '''
    Mien Bac: Xuan 1 -> 3, Ha 4 -> 6, Thu 7 -> 9, Dong 10 -> 12
    Mien Trung: Kho 1 -> 8, Mua 9 -> 12
    Mien Nam: Mua 5 -> 11, Kho 12 -> 4 
    '''

    spring, summer, autumn, winter, dry, rain = 0, 0, 0, 0, 0, 0 
    if north_flag:
        spring = month.isin([1,2,3]).astype(int)
        summer = month.isin([4,5,6]).astype(int)
        autumn = month.isin([7,8,9]).astype(int)
        winter = month.isin([10,11,12]).astype(int)
    elif middle_flag:
        dry = month.isin([1, 2, 3, 4, 5, 6, 7, 8]).astype(int)
        rain = month.isin([9, 10, 11, 12]).astype(int)
    else:
        dry = month.isin([5, 6, 7, 8, 9, 10, 11]).astype(int)
        rain = month.isin([1, 2, 3, 4, 12]).astype(int)
    
    df.insert(df.shape[1], 'spring', spring)
    df.insert(df.shape[1], 'summer', summer)
    df.insert(df.shape[1], 'autumn', autumn)
    df.insert(df.shape[1], 'winter', winter)
    df.insert(df.shape[1], 'dry', dry)
    df.insert(df.shape[1], 'rain', rain)

    print(df.head(n=10))
    return df

def prepare_data(df_ref: pd.DataFrame, n_records: int, lagged_number=7) -> Dict[str, List]:
    '''
        header:
        VN_AQI,CO,PM-10,PM-2-5,SO2,
            mon,tu,wed,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain

        seasons do not changes rapidly, so it just join as the feature of current day

        Lagged: (VN_AQI_7, CO_7,...PM-10_1, PM-2-5_1, SO2_1)
        Current: mon_0,tu_0,wed_0,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain_0
        Label: CO_0, PM-1_0, PM-2-5_0, SO2_0 
    '''
    df = df_ref.copy()
    cols = ['Date','Day_Of_Week','CO','PM-10','PM-2-5','SO2','VN_AQI','mon','tu', 'wed','thu','fri','sat','sun','north','middle','south','spring','summer','autumn','winter','dry','rain']

    data_dict = {
        "X": [], # training data
        "CO": [],
        "PM-10": [],
        "PM-2-5": [],
        "SO2": []
    }

    N = n_records

    df = df[cols]
    df.sort_values(by="Date", ascending=True, inplace=True)
    '''
        get 8 day from the bottom, group 7 days of pollutant from the bottom next to, other info into the rest

        we have total N samples => index from N-1 -> 0 => get (N-1, N-2,..., N-7) N-8 as label
        N - 7 sample

        2:7 == ['VN_AQI', 'CO', 'PM-10', 'PM-2-5', 'SO2']
        7: == ['mon', 'tu', 'wed', 'thu', 'fri', 'sat', 'sun', 'north', 'middle', 'south', 'spring', 'summer', 'autumn', 'winter', 'dry','rain']

        N - 7 - 1 + 1

        [0 -> 4]
        [5 -> 9]
        [10 -> 14]
        [15 -> 19]
        [20 -> 24]
        [25 -> 29]
    '''
    
    for i in range(0, N-7, 1):
        lagged = df.iloc[i:i+7, 2:7].to_numpy(dtype=np.float32)
        rest_features = df.iloc[i+7, 7:].to_numpy(dtype=np.float32)
        Y_CO, Y_PM10, Y_PM25, Y_SO2 = df.iloc[i+7, 2:6].to_numpy(dtype=np.float32)

        lagged = lagged.flatten()
        X = np.concatenate((lagged, rest_features), axis=0)

        data_dict["X"].append(X)
        data_dict["CO"].append(Y_CO)
        data_dict["PM-10"].append(Y_PM10)
        data_dict["PM-2-5"].append(Y_PM25)
        data_dict["SO2"].append(Y_SO2)

    return data_dict
    
def load_data(data_dict: Dict[str, List]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Transform data_dict to numpy_array
    X = np.array(data_dict["X"])
    Y = {
        'CO': np.array(data_dict["CO"]),
        'PM-10': np.array(data_dict["PM-10"]),
        'PM-2-5': np.array(data_dict["PM-2-5"]),
        'SO2': np.array(data_dict["SO2"])
    }
    return X, Y

def predict_next_7_days(df: pd.DataFrame, data_dict: Dict[str, List], ckpt_dir: str):
    latest_day = pd.to_datetime(df['Date'][0])

    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']
    model_params = {
        'CO': f'{ckpt_dir}xgboost_model_CO.pkl',
        'PM-10': f'{ckpt_dir}xgboost_model_PM-10.pkl',
        'PM-2-5': f'{ckpt_dir}xgboost_model_PM-2-5.pkl',
        'SO2': f'{ckpt_dir}xgboost_model_SO2.pkl',
    }

    X_train, Y_train_dict = load_data(data_dict)
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

        # Calculate AQI
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

def get_plot_data(df: pd.DataFrame, dict_of_list: Dict[str, List]):
    """
        dict_of_list['CO'].append(CO_pred)
        dict_of_list['PM-10'].append(PM10_pred)
        dict_of_list['PM-2-5'].append(PM25_pred)
        dict_of_list['SO2'].append(SO2_pred)
    """
    
    # transform date into datetime format
    df.sort_values(by="Date", ascending=False, inplace=True)
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

    # get the latest day
    latest_day = df['Date'].iloc[0]
    
    # crate a list: 19 day before + 1 latest day + 7 days after = 27 days
    day_list = []
    for i in range(1, 20):
        day_list.append(latest_day - pd.Timedelta(days=i))
    day_list.append(latest_day)
    for i in range(1, 8):
        day_list.append(latest_day + pd.Timedelta(days=i))

    # sort by date
    day_list = sorted(day_list)
    date_str_list = [d.strftime("%Y-%m-%d") for d in day_list]

    # create DataFrame
    plot_data = pd.DataFrame(dict_of_list, index=date_str_list)
    plot_data.index.name = 'Date'

    plot_data = plot_data.replace({np.nan: None})

    # Determine the latest day index and the predicted days for plotting task
    latest_idx = date_str_list.index(latest_day.strftime("%Y-%m-%d"))
    predict_start_idx = latest_idx + 1
    predict_end_idx = len(date_str_list)
    
    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']

    # Return primitive 
    return {
        'dates': date_str_list,
        'latest_idx': latest_idx,
        'predict_start_idx': predict_start_idx,
        'predict_end_idx': predict_end_idx,
        'targets': targets,
        'values': plot_data.to_dict(orient='list'),
    }


def visualize(plot_dir: str, df: pd.DataFrame, dict_of_list: Dict[str, List]):
    # Thiết lập style đẹp
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    df.loc[:,'Date'] = pd.to_datetime(df['Date'])

    # Lấy ngày mới nhất
    latest_day = df['Date'].iloc[0]

    # Tạo danh sách ngày: 19 ngày trước + 1 latest + 7 ngày sau = 27 ngày
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
        ax.set_title(f'AQI of {pollutant} through {len(df)+7} days\n'
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

        # Lưu png riêng cho từng pollutant
        png_path = plot_dir + f'aqi_{pollutant}.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Saved plot for {pollutant} at: {png_path}")

        # # Hiển thị (tùy chọn, sẽ hiện từng cái một)
        # Commit these two line because we have been deployed the image on web, if we still show, it will cause an error:
        """
            Exception ignored in: <function Image.__del__ at 0x7040741d63b0>
            Traceback (most recent call last):
            File "/home/lethanhnghia/anaconda3/envs/DATH/lib/python3.10/tkinter/__init__.py", line 4056, in __del__
                self.tk.call('image', 'delete', self.name)
            RuntimeError: main thread is not in main loop
        """
        # plt.show()
        # plt.close(fig)  # Đóng figure để tránh chồng chéo

    print("All 4 separate charts saved successfully!")   

def mongo_pipeline(ckpt_dir, plot_dir=None):
    client = get_mongo_client()
    if client is None:
        print("[ERROR]: Failed to connect to MongoDB.")
        return
    print("[INFO]: Connected to MongoDB.")
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    date = 20
    try:
        raw_df = crawl_data(date=date)
    except Exception as e:
        print(f"[WARNING]: Cannot get data from the server, reuse the data in MongoDB to temperally render to web viewer!") 
        raw_df = None
        
    if raw_df is not None:
        df = drop_col(raw_df)
        df = feature_engineering_and_preprocessing(df, region="north")
    else:
        records = list(collection.find({}, {"_id": 0}))   # get all the records in the database, then remove the _id field of MongoDB
        df = pd.DataFrame(records)

    latest_day = df['Date'].iloc[0]
    if collection.find_one({
        "Date": latest_day
    }):
        print(f"[INFO]: Data for {latest_day} already exists in MongoDB. Do not update the database")
    else:
        print(f"[INFO]: Data for {latest_day} does not exist in MongoDB. Updating the database.")
        records = df.to_dict(orient='records')     # create data as: [{row1}, {row2}, {row3},...,{rowN}]

        # Store records into MongoDB as many record => optimize for the large document, but we also query and concate them
        collection.insert_many(records)
    

    data_dict = prepare_data(df, date)

    dict_of_list = predict_next_7_days(df=df, data_dict=data_dict, ckpt_dir=ckpt_dir)
    # visualize(plot_dir=plot_dir, df=df, dict_of_list=dict_of_list)
    
    return get_plot_data(df=df, dict_of_list=dict_of_list)

if __name__ == "__main__":
    # mongo_pipeline(ckpt_dir='./model_nghia/XGBoost/ckpts/', plot_dir='./static/plots/mongo/')
    mongo_pipeline(ckpt_dir='./model_nghia/XGBoost/ckpts/')