import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
MODEL_PATH = 'kmeans_aqi_model.pkl'
SCALER_PATH = 'minmax_scaler.pkl'
FEATURES = ['CO', 'NO2', 'PM-10', 'PM-2-5', 'SO2']

# MAPPING KẾT QUẢ (QUAN TRỌNG: Bạn phải tự chỉnh lại dựa trên Heatmap của bạn)
# Ví dụ: Nếu cụm 0 có chỉ số thấp nhất -> Là "An toàn"
# Nếu cụm 2 có chỉ số cao nhất -> Là "Nguy hại"
CLUSTER_LABELS = {
    0: {"name": "Rất xấu", "color": "purple", "msg": "Không khí chất lượng rất xấu, chỉ nên ra đường nếu cần thiết."},
    1: {"name": "Tốt", "color": "green", "msg": "Chất lượng không khí tốt!"},
    2: {"name": "Nguy hại", "color": "brown", "msg": "Cảnh báo sức khỏe khẩn cấp!"},
    3: {"name": "Trung bình", "color": "yellow", "msg": "Chất lượng không khí trung bình."},
    4: {"name": "Xấu", "color": "red", "msg": "Không khí xấu, hạn chế ra đường nếu không cần thiết."},
    5: {"name": "Kém", "color": "orange", "msg": "Không khí chất lượng kém."}
}

# --- LOAD MODEL ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Không tìm thấy file model. Hãy chạy Bước 1 để lưu model.pkl và scaler.pkl trước!")
    st.stop()

# --- GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="Dự báo AQI", page_icon="")

st.title("Hệ thống Phân vùng Chất lượng Không khí")
st.markdown("Nhập các chỉ số cảm biến để phân loại mức độ ô nhiễm.")

# Chia cột để giao diện đẹp hơn
col1, col2 = st.columns(2)

input_data = {}

with col1:
    st.subheader("Nhập dữ liệu cảm biến")
    # Tạo input cho từng đặc trưng
    for feat in FEATURES:
        # Giá trị mặc định là min/max thực tế (ước lượng) để dễ nhập
        input_data[feat] = st.number_input(f"Chỉ số {feat}", min_value=0.0, format="%.2f")

with col2:
    st.subheader("Kết quả Phân tích")
    
    # Nút bấm dự báo
    if st.button("Phân tích ngay", use_container_width=True):
        # 1. Chuẩn bị dữ liệu
        input_df = pd.DataFrame([input_data]) # Tạo DataFrame 1 dòng
        
        # 2. Scale dữ liệu (Bắt buộc dùng scaler đã train)
        input_scaled = scaler.transform(input_df.values)
        
        # 3. Dự báo
        cluster_id = model.predict(input_scaled)[0]
        result = CLUSTER_LABELS.get(cluster_id, {"name": "Unknown", "color": "gray", "msg": "..."})

        # 4. Hiển thị kết quả
        st.success(f"Kết quả phân cụm: **Nhóm {cluster_id}**")
        
        # Hiển thị thẻ màu cảnh báo
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {result['color']}; color: white; text-align: center;">
                <h2 style="margin:0;">{result['name']}</h2>
                <p style="margin:5px;">{result['msg']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # 5. Vẽ biểu đồ Radar (Spider Chart) để xem lệch về chất nào
        # (Chuẩn bị dữ liệu vẽ)
        categories = FEATURES
        values = input_scaled.flatten().tolist()
        values += values[:1] # Khép kín vòng tròn
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color=result['color'], alpha=0.25)
        ax.plot(angles, values, color=result['color'], linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Biểu đồ mức độ ô nhiễm (Normalized)", size=10, y=1.1)
        
        st.pyplot(fig)

# --- PHẦN FOOTER ---
st.markdown("---")
st.caption("BTL Khai phá Dữ liệu - K-Means Clustering Baseline Demo")