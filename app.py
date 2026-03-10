import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, send_from_directory, flash, redirect, url_for, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

try: 
    # from crawl import crawl_data
    # from data_processor import drop_col, feature_engineering_and_preprocessing, prepare_data
    # from plot_prediction import predict_next_7_days, visualize
    from mongo_pipeline import mongo_pipeline, set_last_run_time, get_last_run_time
except ImportError as e:
    print(f"Friend's Pipeline Import Error: {e}")

try:
    from ai_engine.aqi_service import AQIPredictor
    print("[MY MODULE] Đang khởi tạo AI Engine...")
    ai_bot = AQIPredictor(model_path='./ai_engine/weights/best_aqi_model_dual.pth')
    print("[MY MODULE] AI Engine sẵn sàng!")
except Exception as e:
    print(f"[MY MODULE] Lỗi khởi tạo AI Engine: {e}")
    ai_bot = None

# --- CONFIG CHUNG ---
data_dir = './data/'
ckpts_dir = './model_nghia/XGBoost/ckpts/'
plots_dir = './static/plots/'
plot_dir_mongo = './static/plots/mongo/'

app = Flask(__name__)
app.secret_key = 'supersecretkey'

LAST_RUN_TIME = get_last_run_time()
PLOT_DICT = None
scheduler = None
PIPELINE_STATUS = "idle"   # idle | running | done | error

# ==========================
# KHU VỰC 1: LOGIC CỦA NGHĨA
# ==========================        

def run_pipeline():
    """Logic chạy pipeline cũ (XGBoost + Static Plots)"""
    global LAST_RUN_TIME, PLOT_DICT, PIPELINE_STATUS

    print("\n--- STARTING AUTOMATIC DATA PIPELINE ---")
    PIPELINE_STATUS = "running"

    try:
        PLOT_DICT = mongo_pipeline(ckpt_dir=ckpts_dir, plot_dir=plot_dir_mongo)

        PIPELINE_STATUS = "done"
        LAST_RUN_TIME = set_last_run_time()
        
        print(f"--- PIPELINE COMPLETE. LAST RAN: {LAST_RUN_TIME} ---")
    except Exception as e:
        print(f"Error while running Pipeline: {e}")
        PIPELINE_STATUS = "error"

def start_scheduler():
    global scheduler

    scheduler = BackgroundScheduler()
    # Chạy pipeline mỗi ngày lúc 5h sáng
    scheduler.add_job(func=run_pipeline, trigger='cron', hour=5, minute=0, id='daily_crawl')
    
    # Có thể thêm job chạy ngay khi khởi động để test (bỏ comment nếu cần)
    # scheduler.add_job(func=run_pipeline, trigger='date', run_date=datetime.now() + timedelta(seconds=5), id='initial_run')
    
    try:
        scheduler.start()
    except Exception as e:
        print(f"Scheduler Error: {e}")

@app.route('/')
def index():
    pollutant_names = ['CO', 'PM-10', 'PM-2-5', 'SO2']

    return render_template(
        'index.html', 
        pollutant_names=pollutant_names,
        last_run=LAST_RUN_TIME, 
        pipeline_status=PIPELINE_STATUS
    )

# Update run_now to support AJAX request (POST request) - which do not need to re-load the whole page
@app.route('/run_now', methods=['POST'])
def run_now():
    """
        This function need to return the data with JSON format instead of redirect and flash.
        If using redirect and flash, so we need to re-load the whole page...
        But the run_pipeline function is running in background, so we just need to update when it done => Polling the pipeline status every x seconds.
    """
    global scheduler, PIPELINE_STATUS
    # Do not schedule multiple runs at one time
    if PIPELINE_STATUS == "running":
        return jsonify({
            "status": "warning",
            "message": "Pipeline is running. Please wait a seconds",
        })

    PIPELINE_STATUS = "running"

    scheduler.add_job(
        func=run_pipeline, 
        trigger='date', 
        id='manual_run',
        run_date=datetime.now() + timedelta(seconds=1),
        replace_existing=True
    )
    return jsonify({
        "status": "success",
        "message": "Pipeline collect and process data started!"
    })

@app.route('/pipeline_status', methods=["POST", "GET"])
def pipeline_status():
    global PIPELINE_STATUS, LAST_RUN_TIME
    return jsonify({
        'PIPELINE_STATUS': PIPELINE_STATUS,
        'LAST_RUN_TIME': LAST_RUN_TIME,        
    })

@app.route('/plot_dict')
def plot_dict():
    global PLOT_DICT
    if PLOT_DICT is None:
        return jsonify({
            'status': 'error', 
            'message': 'No plot data available yet!'
        }, 500)

    return jsonify({
        "plot_dict": PLOT_DICT
    })

@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(os.path.join(app.static_folder, 'plots/mongo'), filename)


# ==========================
# KHU VỰC 2: LOGIC CỦA DK
# ========================
def get_history_data_for_ai(province_name):
    """Đọc dữ liệu lịch sử từ file CSV trong folder ai_engine"""
    json_path = './ai_engine/data/origin/dataset_info.json'
    data_dir = './ai_engine/data/origin/'
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file info tại: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)
    
    target_file = None
    for key, info in info_data.items():
        if info.get('province') == province_name:
            target_file = info.get('file_name')
            break
            
    if not target_file:
        raise ValueError(f"Không tìm thấy dữ liệu cho tỉnh {province_name}")

    full_path = os.path.join(data_dir, target_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {full_path}")

    df = pd.read_csv(full_path)
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(by='Date', ascending=True)

    print(f"\nDEBUG DATA FETCH")
    print(f"Tỉnh yêu cầu: {province_name}")
    print(f"File nguồn:   {target_file}")
    print(f"Dữ liệu từ:   {df['Date'].min().strftime('%Y-%m-%d')} -> {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Tổng số dòng: {len(df)}")
    print(f"5 dòng cuối cùng (Input cho Model):")
    print(df[['Date', 'VN_AQI']].tail(5).to_string(index=False))
    print("="*40 + "\n")
    # ------------------------------------------

    return df.tail(40)

# 1. Route Giao diện Bản đồ
@app.route('/map')
def map_interface():
    return render_template('map.html')

# 2. API Dự báo
@app.route('/api/predict_advanced', methods=['POST'])
def predict_advanced():
    if not ai_bot:
        return jsonify({'status': 'error', 'message': 'AI Engine chưa được khởi tạo!'}), 500

    try:
        data = request.json
        province = data.get('province')
        print(f"[MY MODULE] Request dự báo: {province}")

        df_history = get_history_data_for_ai(province)
        forecast_result = ai_bot.predict_next_7_days(df_history, province)
        history_to_show = df_history.tail(14).copy()
        history_list = []
        for _, row in history_to_show.iterrows():
            history_list.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'aqi': row['VN_AQI']
            })

        return jsonify({
            'status': 'success',
            'province': province,
            'history': history_list,
            'forecast': forecast_result
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # run_pipeline() # <-- Xóa hoặc comment dòng này
    start_scheduler() # Khởi động trình lập lịch
    
    # Lấy PORT từ biến môi trường của Render (mặc định 10000)
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)