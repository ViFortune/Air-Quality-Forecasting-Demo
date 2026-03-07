import requests
import json
import csv
import datetime
from pathlib import Path

def crawl_data(dir, date=30, type=0):
    '''
        :param type: 0 if get average value by day, 1 to get exactly value by hours (max 26 value) 
    '''

    # define the data folder
    

    # define url
    url = f"https://envisoft.gov.vn/eip/default/call/json/get_aqi_data%3Fdate%3D{date}%26aqi_type%3D{type}"

    # get station_info object
    with open("JSON/station_info.json", "r", encoding="utf-8") as f:
        station_dict = json.load(f)
    
    # get number of stations
    # num_station = station_dict["num_stations"]
    num_station = 1


    for i in range(num_station):

        # station_id = station_dict[str(i)]["id"]
        # alias = station_dict[str(i)]["alias"]
        
        station_id = 31390908889087377344742439468
        alias = "HN_KDT_KK" # Ha Noi, Khuat Duy Tien KK
        add_col = "CO,NO2,O3,PM-10,PM-2-5,SO2"
        station_name = "Ha Noi: Khuat Duy Tien KK"
        components = list(add_col.split(","))

        data = {
            # "sEcho": "1",
            # "iColumns": "27",
            # "sColumns": ",,,,,,,,,,,,,,,,,,,,,,,,,,",
            # "iDisplayStart": "0",
            # "iDisplayLength": f"{date}",
            # "mDataProp_0": "0",
            # "sSearch_0": "",
            # "bRegex_0": "false",
            # "bSearchable_0": "true",
            # "mDataProp_1": "1",
            # "sSearch_1": "",
            # "bRegex_1": "false",
            # "bSearchable_1": "true",
            # "mDataProp_2": "2",
            # "sSearch_2": "",
            # "bRegex_2": "false",
            # "bSearchable_2": "true",
            # "mDataProp_3": "3",
            # "sSearch_3": "",
            # "bRegex_3": "false",
            # "bSearchable_3": "true",
            # "mDataProp_4": "4",
            # "sSearch_4": "",
            # "bRegex_4": "false",
            # "bSearchable_4": "true",
            # "mDataProp_5": "5",
            # "sSearch_5": "",
            # "bRegex_5": "false",
            # "bSearchable_5": "true",
            # "mDataProp_6": "6",
            # "sSearch_6": "",
            # "bRegex_6": "false",
            # "bSearchable_6": "true",
            # "mDataProp_7": "7",
            # "sSearch_7": "",
            # "bRegex_7": "false",
            # "bSearchable_7": "true",
            # "mDataProp_8": "8",
            # "sSearch_8": "",
            # "bRegex_8": "false",
            # "bSearchable_8": "true",
            # "mDataProp_9": "9",
            # "sSearch_9": "",
            # "bRegex_9": "false",
            # "bSearchable_9": "true",
            # "mDataProp_10": "10",
            # "sSearch_10": "",
            # "bRegex_10": "false",
            # "bSearchable_10": "true",
            # "mDataProp_11": "11",
            # "sSearch_11": "",
            # "bRegex_11": "false",
            # "bSearchable_11": "true",
            # "mDataProp_12": "12",
            # "sSearch_12": "",
            # "bRegex_12": "false",
            # "bSearchable_12": "true",
            # "mDataProp_13": "13",
            # "sSearch_13": "",
            # "bRegex_13": "false",
            # "bSearchable_13": "true",
            # "mDataProp_14": "14",
            # "sSearch_14": "",
            # "bRegex_14": "false",
            # "bSearchable_14": "true",
            # "mDataProp_15": "15",
            # "sSearch_15": "",
            # "bRegex_15": "false",
            # "bSearchable_15": "true",
            # "mDataProp_16": "16",
            # "sSearch_16": "",
            # "bRegex_16": "false",
            # "bSearchable_16": "true",
            # "mDataProp_17": "17",
            # "sSearch_17": "",
            # "bRegex_17": "false",
            # "bSearchable_17": "true",
            # "mDataProp_18": "18",
            # "sSearch_18": "",
            # "bRegex_18": "false",
            # "bSearchable_18": "true",
            # "mDataProp_19": "19",
            # "sSearch_19": "",
            # "bRegex_19": "false",
            # "bSearchable_19": "true",
            # "mDataProp_20": "20",
            # "sSearch_20": "",
            # "bRegex_20": "false",
            # "bSearchable_20": "true",
            # "mDataProp_21": "21",
            # "sSearch_21": "",
            # "bRegex_21": "false",
            # "bSearchable_21": "true",
            # "mDataProp_22": "22",
            # "sSearch_22": "",
            # "bRegex_22": "false",
            # "bSearchable_22": "true",
            # "mDataProp_23": "23",
            # "sSearch_23": "",
            # "bRegex_23": "false",
            # "bSearchable_23": "true",
            # "mDataProp_24": "24",
            # "sSearch_24": "",
            # "bRegex_24": "false",
            # "bSearchable_24": "true",
            # "mDataProp_25": "25",
            # "sSearch_25": "",
            # "bRegex_25": "false",
            # "bSearchable_25": "true",
            # "mDataProp_26": "26",
            # "sSearch_26": "",
            # "bRegex_26": "false",
            # "bSearchable_26": "true",
            # "sSearch": "",
            # "bRegex": "false",
            # "station_id": f"{station_id}",
            # "from_date": "",
            # "added_columns": "Benzen,CH4,CO,Compass,EthylBenzen,HC,HCL,MMHC,Mp-Xylen,NMHC,NO,NO2,NO2,NOx,O3,Oxylen,PM-10,PM-2-5,PX2-1,SO2,SO2,THC,Toluren,WinDir+(sai)"
            "sEcho": "1",
            "iColumns": "9",
            "sColumns": ",,,,,,,,",
            "iDisplayStart": "0",
            "iDisplayLength": "20",
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
                with open(f"{dir}{alias}.csv", "w", encoding='utf-8') as opened_csv:
                    writer = csv.writer(opened_csv)
                    # writer.writerow(["STT","Date","VN_AQI","Benzen","CH4","CO","Compass","EthylBenzen","HC","HCL","MMHC","Mp-Xylen","NMHC","NO","NO2","NO2","NOx","Oxylen","PM-10","PM-2-5","PX2-1","SO2","SO2","THC","Toluren","WinDir+(sai)"])
                    writer.writerow(["STT","Date","VN_AQI"] + components)   # header of csv file

                    result = response.json()
                    aaData = result["aaData"] # Array of array

                    for data_row in aaData:
                        writer.writerow(data_row)

                # print(f"{station_dict[str(i)]['station_name']} has been recorded with number of samples is {result['iTotalRecords']-1}\n")
                print(f"{station_name} has been recorded with number of samples is {result['iTotalRecords']-1}\n")

            except ValueError:
                print("Responsed value is not JSON format, store it as text")
                with open(f"{dir}{alias}.csv", "w", encoding="utf-8") as opened_text:
                    opened_text.write(response.text)

        else:
            print(f"Error code: {response.status_code}!\n")
            print(response.text)


if __name__ == '__main__':
    crawl_data(dir='data/', date=30)