import torch
import joblib
import json
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from lstm_model.model import DualEmbeddingBiLSTM
from lstm_model.configs import cfg

class AQIPredictor:
    def __init__(self, model_path='./ai_engine/weights/best_aqi_model_dual.pth'):
        self.device = torch.device('cpu')
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        try:
            self.station_enc = joblib.load(os.path.join(base_path, 'encoders/station_encoder.pkl'))
            self.region_enc = joblib.load(os.path.join(base_path, 'encoders/region_encoder.pkl'))
            self.scaler = joblib.load(os.path.join(base_path, 'normalized_data/global_scaler.pkl'))
            
            json_path = os.path.join(base_path, 'data/origin/dataset_info.json')
            self.province_map = self._build_province_mapping(json_path)
        except FileNotFoundError as e:
            print(f"Missing resources: {e}")
            raise

        num_stations = len(self.station_enc.classes_)
        num_regions = len(self.region_enc.classes_)
        input_dim = 23 
        
        self.model = DualEmbeddingBiLSTM(
            config=cfg,
            num_stations=num_stations,
            num_regions=num_regions,
            input_dim=input_dim
        ).to(self.device)
        
        if os.path.exists(model_path):
             checkpoint_path = model_path
        else:
             checkpoint_path = os.path.join(base_path, 'weights/best_aqi_model_dual.pth')

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("AQI Predictor ready!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def _build_province_mapping(self, json_path):
        if not os.path.exists(json_path):
            return {}
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mapping = {}
        for key, info in data.items():
            prov = info.get('province')
            filename = info.get('file_name')
            if prov and filename:
                station_id = '_'.join(filename.split('_')[:-1])
                if prov not in mapping:
                    mapping[prov] = {'region': info.get('region'), 'default_station': station_id}
        return mapping

    def _inverse_transform_output(self, scaled_aqi):
        try:
            n_features = self.scaler.n_features_in_
            dummy_row = np.zeros((1, n_features))
            
            if hasattr(self.scaler, 'feature_names_in_'):
                col_names = list(self.scaler.feature_names_in_)
                try:
                    aqi_idx = col_names.index('VN_AQI')
                except ValueError:
                    aqi_idx = 0
            else:
                aqi_idx = 0
                
            dummy_row[0, aqi_idx] = scaled_aqi
            original_row = self.scaler.inverse_transform(dummy_row)

            return original_row[0, aqi_idx]
            
        except Exception as e:
            print(f"Inverse scaling error: {e}")
            return scaled_aqi

    def _add_engineered_features(self, df):
        df = df.copy()
        
        # Data Cleaning
        numeric_cols = ['VN_AQI', 'CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].bfill().ffill()
        
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
        df['hour'] = df['Date'].dt.hour
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        for col in ['VN_AQI', 'PM-2-5', 'PM-10']:
            if col in df.columns:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_7'] = df[col].shift(7)
        
        for col in ['VN_AQI', 'PM-2-5']:
            if col in df.columns:
                df[f'{col}_roll_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
        
        return df.bfill().ffill()

    def preprocess_sequence(self, df_history):
        df_processed = self._add_engineered_features(df_history)
        try:
            if hasattr(self.scaler, 'feature_names_in_'):
                cols_expected_by_scaler = self.scaler.feature_names_in_
            else:
                cols_expected_by_scaler = [
                    'VN_AQI', 'CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2',
                    'VN_AQI_lag_1', 'VN_AQI_lag_7', 
                    'PM-2-5_lag_1', 'PM-2-5_lag_7', 
                    'PM-10_lag_1', 'PM-10_lag_7', 
                    'VN_AQI_roll_mean_7', 'PM-2-5_roll_mean_7'
                ]

            missing_cols = [c for c in cols_expected_by_scaler if c not in df_processed.columns]
            if missing_cols:
                for c in missing_cols: df_processed[c] = 0.0
            
            data_window = df_processed.tail(24).copy()
            data_to_scale = data_window[cols_expected_by_scaler]
            scaled_values = self.scaler.transform(data_to_scale)
            
            df_scaled = pd.DataFrame(scaled_values, columns=cols_expected_by_scaler, index=data_window.index)
            
            for col in data_window.columns:
                if col not in df_scaled.columns:
                    df_scaled[col] = data_window[col]

            model_features = [
                'CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2', 
                'hour', 'hour_sin', 'hour_cos', 
                'day_of_week', 'month', 'quarter', 
                'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'VN_AQI_lag_1', 'VN_AQI_lag_7', 
                'PM-2-5_lag_1', 'PM-2-5_lag_7', 
                'PM-10_lag_1', 'PM-10_lag_7', 
                'VN_AQI_roll_mean_7', 'PM-2-5_roll_mean_7'
            ]
            
            final_data = df_scaled[model_features].values
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

        return torch.tensor(final_data, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _update_history_for_recursive(self, current_df, predicted_aqi):
        last_row = current_df.iloc[-1].copy()
        next_date = last_row['Date'] + timedelta(days=1)
        new_row = last_row.copy()
        new_row['Date'] = next_date
        new_row['VN_AQI'] = predicted_aqi 
        
        return pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    def predict_next_7_days(self, df_history, province_name):
        if province_name not in self.province_map:
            print(f"❌ Province not found: {province_name}")
            return []
            
        info = self.province_map[province_name]
        station_id = info['default_station']
        region = info['region']
        
        print(f"DEBUG MAPPING")
        print(f"Tỉnh:   {province_name}")
        print(f"Trạm:   {station_id}")
        print(f"Vùng:   {region}")
        print(f"Input:  {len(df_history)} records")
        print("-" * 30)
        # ----------------------------------------
        
        try:
            try: s_val = self.station_enc.transform([station_id])[0]
            except: s_val = 0 
            try: r_val = self.region_enc.transform([region])[0]
            except: r_val = 0
            s_idx = torch.tensor([s_val], dtype=torch.long).to(self.device)
            r_idx = torch.tensor([r_val], dtype=torch.long).to(self.device)
        except: return []

        future_predictions = []
        running_history = df_history.copy()
        running_history['Date'] = pd.to_datetime(running_history['Date'], dayfirst=True)

        print(f"Forecasting 7 days for: {province_name}")

        for i in range(7):
            input_tensor = self.preprocess_sequence(running_history)
            if input_tensor is None: break

            with torch.no_grad():
                scaled_pred = self.model(input_tensor, s_idx, r_idx).item()
            
            real_pred = self._inverse_transform_output(scaled_pred)
            real_pred = max(0, real_pred)
            
            last_date = running_history.iloc[-1]['Date']
            next_date = last_date + timedelta(days=1)
            
            future_predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "aqi": round(real_pred, 2)
            })
            
            running_history = self._update_history_for_recursive(running_history, real_pred)

        return future_predictions