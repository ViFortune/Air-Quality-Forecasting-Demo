import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch

"""
    "0": {
        "file_name": "HN_NguyenVanCu_28560877461938780203765592307.csv",
        "region": "north",
        "n_records": 30,
        "header": [
            "Date",
            "VN_AQI",
            "CO",
            "NO2",
            "PM-10",
            "PM-2-5",
            "SO2"
        ]
    }
"""

def drop_col(data_dir):
    with open(data_dir+'dataset_info.json', "r", encoding='utf-8') as json_info:
        info = json.load(json_info)

    for i in range(len(info)):
        csv_path = info[str(i)]["file_name"]
        df = pd.read_csv(data_dir+csv_path, delimiter=',')
        # df = df.T.drop_duplicates().T
        df = df.drop(columns=["STT"], axis=1)
        df = df.drop(columns=['NO2'], axis=1)
        df = df.drop(columns=['O3'], axis=1)
        
        for attr in df.columns.to_list():
            arr: np
            arr = df[attr].to_numpy()
            miss_rate = np.where(arr == '-')[0].shape[0]/df.shape[0]
            
            if miss_rate > 0.9:
                print(f"Remove column [{attr}] with [{miss_rate*100:.5}%] missing rate")
                df = df.drop(columns=[attr], axis=1)

        df.to_csv(data_dir+'processed_'+csv_path, sep=',', index=False, encoding='utf-8')
        return

def feature_engineering_and_preprocessing(data_dir):
    '''
        This function is responsible for adding nefw columns to our csv files
        Adding some field to robust our 
    '''
    json_file = data_dir + "dataset_info.json"
    with open(json_file, "r", encoding='utf-8') as jf:
        jobj = json.load(jf)
    for i in range(len(jobj)):
        file_name = jobj[str(i)]["file_name"]
        csv_obj = pd.read_csv(data_dir+'processed_'+file_name, delimiter=',')
        csv_obj = csv_obj.drop_duplicates()
        csv_obj = csv_obj.replace('-', pd.NA)
        
        # Transform date into DOW
        csv_obj['Date'] = pd.to_datetime(csv_obj["Date"], format='%d/%m/%Y')
        month = csv_obj['Date'].dt.month
        # print(csv_obj['Date'].to_list())
        DOW = csv_obj['Date'].dt.day_of_week
        csv_obj.insert(1, 'Day_Of_Week', DOW)

        csv_obj.insert(csv_obj.shape[1], 'mon', csv_obj["Day_Of_Week"].isin([0]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'tu', csv_obj["Day_Of_Week"].isin([1]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'wed', csv_obj["Day_Of_Week"].isin([2]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'thu', csv_obj["Day_Of_Week"].isin([3]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'fri', csv_obj["Day_Of_Week"].isin([4]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'sat', csv_obj["Day_Of_Week"].isin([5]).astype(int))
        csv_obj.insert(csv_obj.shape[1], 'sun', csv_obj["Day_Of_Week"].isin([6]).astype(int))

        # Adding region addtribute: north, middle, south
        region = jobj[str(i)]["region"]
        north_flag, middle_flag, south_flag = 0, 0, 0
        if region == 'north':
            north_flag = 1
        
        elif region == 'middle':
            middle_flag = 1
            
        else:
            south_flag = 1

        csv_obj.insert(csv_obj.shape[1], 'north', north_flag)
        csv_obj.insert(csv_obj.shape[1], 'middle', middle_flag)
        csv_obj.insert(csv_obj.shape[1], 'south', south_flag)
        
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
        
        csv_obj.insert(csv_obj.shape[1], 'spring', spring)
        csv_obj.insert(csv_obj.shape[1], 'summer', summer)
        csv_obj.insert(csv_obj.shape[1], 'autumn', autumn)
        csv_obj.insert(csv_obj.shape[1], 'winter', winter)
        csv_obj.insert(csv_obj.shape[1], 'dry', dry)
        csv_obj.insert(csv_obj.shape[1], 'rain', rain)

        # print(csv_obj.head(3))

        # Transform into season
        # print(csv_obj.isnull().sum()/csv_obj.shape[0])

        print(csv_obj.isnull().sum()/csv_obj.shape[0])

        # train_part = train_part.drop(columns=['Date', 'Day_Of_Week'], axis=1)
        # test_part = test_part.drop(columns=['Date', 'Day_Of_Week'], axis=1)

        '''
            Before normalizing, we need to split dataset into trainset and testset maybe
        '''
        # train_part, test_part, scaler_subict = normalize(train_part, test_part, file_name)
        # scaler_dict[str(i)] = scaler_subict

        csv_obj.to_csv(data_dir+'processed_'+file_name, sep=',', index=False)
    return

'''
    header:
    VN_AQI,CO,PM-10,PM-2-5,SO2,
        mon,tu,wed,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain

    seasons do not changes rapidly, so it just join as the feature of current day

    Lagged: (VN_AQI_7, CO_7,...PM-10_1, PM-2-5_1, SO2_1)
    Current: mon_0,tu_0,wed_0,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain_0
    Label: CO_0, PM-1_0, PM-2-5_0, SO2_0 
'''

def prepare_data(dir, dst_pt, lagged_number=7):

    cols = ['Date','Day_Of_Week','CO','PM-10','PM-2-5','SO2','VN_AQI','mon','tu', 'wed','thu','fri','sat','sun','north','middle','south','spring','summer','autumn','winter','dry','rain']

    json_file = dir + "dataset_info.json"
    with open(json_file, "r", encoding='utf-8') as j:
        jobj = json.load(j)

    data_dict = {
        "X": [], # training data
        "CO": [],
        "PM-10": [],
        "PM-2-5": [],
        "SO2": []
    }

    for index in range(len(jobj)):
        fname = jobj[str(index)]["file_name"]
        N = jobj[str(index)]["n_records"]

        csv_obj = pd.read_csv(dir+'processed_'+fname, sep=',', encoding='utf-8')
        csv_obj = csv_obj[cols]
        csv_obj.sort_values(by="Date", ascending=True, inplace=True)
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
            lagged = csv_obj.iloc[i:i+7, 2:7].to_numpy(dtype=np.float32)
            rest_features = csv_obj.iloc[i+7, 7:].to_numpy(dtype=np.float32)
            Y_CO, Y_PM10, Y_PM25, Y_SO2 = csv_obj.iloc[i+7, 2:6].to_numpy(dtype=np.float32)

            lagged = lagged.flatten()
            X = np.concatenate((lagged, rest_features), axis=0)

            data_dict["X"].append(X)
            data_dict["CO"].append(Y_CO)
            data_dict["PM-10"].append(Y_PM10)
            data_dict["PM-2-5"].append(Y_PM25)
            data_dict["SO2"].append(Y_SO2)

    torch.save(data_dict, dst_pt)
    return

if __name__ == '__main__':
    drop_col('./data/')
    feature_engineering_and_preprocessing('./data/')
    prepare_data('./data/', './data/real_time_v1.pt')