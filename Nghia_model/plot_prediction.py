import matplotlib.pyplot as plt
import numpy as np
import joblib # Dùng để lưu và tải mô hình
from train import load_data


'''
    In this file, we handle the task which predict day by day and plot them into 1 plot with the ground truth, then the second task is that bases on the previous prediction to predict the next, next day to forecast.... 
'''


def main():
    targets = ['CO', 'PM-10', 'PM-2-5', 'SO2']
    model_params = {
        'CO': 'xgboost_model_CO.pkl',
        'PM-10': 'xgboost_model_PM-10.pkl',
        'PM-2-5': 'xgboost_model_PM-2-5.pkl',
        'SO2': 'xgboost_model_SO2.pkl',
    }
    real_time_path = 'real-time-data/real_time.pt'
    X_train, Y_train_dict = load_data(real_time_path)

    # First we should do with the PM-10 for the first demo
    # But the test set is the whole test sample from all station, so we have to modified the specific station
    PM10_list = []
    PM10_pred_list = []
    model = joblib.load('xgboost_model_PM-10.pkl')
    for example_idx in range(len(X_train)):
        X_PM10_lagged = np.array([X_train[example_idx]])
        if example_idx == 0:
            PM10_list.extend([X_train[example_idx][i] for i in range(1, 32, 5)])
            PM10_pred_list.extend([np.array(0, dtype=np.float32) for i in range(1, 32, 5)])

        PM10_pred = model.predict(X_PM10_lagged)
        PM10_pred_list.append(PM10_pred[0])
        PM10_list.append(Y_train_dict['PM-10'][example_idx])
    
    plt.figure(figsize=(7,3))
    plt.plot(np.asarray(range(1, len(PM10_list)+1, 1)), np.asarray(PM10_list), color='blue', marker='o', markersize=3)
    plt.plot(np.asarray(range(1, len(PM10_pred_list)+1, 1)), np.asarray(PM10_pred_list), color='green', marker='o', markersize=3)

    plt.plot()
    plt.show()

    

if __name__ == '__main__':
    main()