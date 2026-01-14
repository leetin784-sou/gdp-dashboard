import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# =========================
# 🌑 DARK THEME CUSTOM CSS
# =========================
st.set_page_config(page_title="Hệ thống Cấp cứu AI Pro", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    /* Nền ứng dụng tối */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Sidebar chuyên nghiệp */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Thẻ thông tin (Cards) */
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Tùy chỉnh input (Text/Number/Select) */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #1e293b !important;
        border-radius: 8px;
    }

    /* Nút bấm Phân loại */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }

    /* Kết quả Phân loại */
    .triage-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        margin-top: 20px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# ⚙️ LOGIC HỖ TRỢ
# =========================
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def calculate_ews(hr, rr, sbp, temp, spo2):
    score = 0
    if hr > 115 or hr < 45: score += 3
    if rr > 26 or rr < 10: score += 3
    if sbp < 90: score += 3
    if spo2 < 92: score += 3
    if temp > 38.5 or temp < 35.5: score += 1
    return score

# =========================
# 📟 SIDEBAR DASHBOARD
# =========================
with st.sidebar:
    st.title("🏥 Triage Console")
    st.write(f"📅 **{datetime.now().strftime('%d/%m/%Y | %H:%M')}**")
    st.divider()
    
    if st.session_state["logs"]:
        df_logs = pd.DataFrame(st.session_state["logs"])
        st.metric("Tổng ca tiếp nhận", len(df_logs))
        red_count = len(df_logs[df_logs['Phân loại'].str.contains("ĐỎ")])
        st.error(f"🚨 Ca Nguy kịch: {red_count}")
    
    st.divider()
    if st.button("🗑️ Xóa bộ nhớ"):
        st.session_state["logs"] = []
        st.rerun()

# =========================
# 🏥 GIAO DIỆN NHẬP LIỆU
# =========================
st.title("🚑 Hệ thống Phân loại Cấp cứu - Dark Mode")

tab1, tab2 = st.tabs(["📑 Tiếp nhận Bệnh nhân", "📊 Thống kê Khoa"])

with tab1:
    with st.form("dark_triage_form"):
        col1, col2, col3 = st.columns(3, gap="large")
        
        with col1:
            st.subheader("🩺 Sinh hiệu")
            age = st.number_input("Tuổi", 0, 120, 30)
            hr = st.number_input("Mạch (BPM)", 20, 250, 80)
            sbp = st.number_input("HA Tâm thu (mmHg)", 40, 260, 120)
            spo2 = st.slider("SpO₂ (%)", 60, 100, 98)
            rr = st.number_input("Nhịp thở", 5, 60, 18)
            temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 36.6)

        with col2:
            st.subheader("🧠 Thần kinh & Đau")
            gcs_e = st.selectbox("Mắt (E)", [4, 3, 2, 1])
            gcs_v = st.selectbox("Lời nói (V)", [5, 4, 3, 2, 1])
            gcs_m = st.selectbox("Vận động (M)", [6, 5, 4, 3, 2, 1])
            gcs_total = gcs_e + gcs_v + gcs_m
            
            pain = st.select_slider("Mức độ đau (VAS)", options=range(11), value=0)

        with col3:
            st.subheader("🚩 Cảnh báo nhanh")
            chest_pain = st.checkbox("Đau ngực cấp")
            dyspnea = st.checkbox("Khó thở cấp")
            altered_mental = st.checkbox("Lú lẫn / Kích động")
            trauma = st.checkbox("Chấn thương nặng")
            
        submit = st.form_submit_button("PHÂN LOẠI NGAY")

    if submit:
        # --- LOGIC PHÂN LOẠI ---
        ews = calculate_ews(hr, rr, sbp, temp, spo2)
        flags = []
        if gcs_total <= 8: flags.append("Hôn mê")
        if spo2 < 90: flags.append("SpO2 cực thấp")
        if sbp < 90: flags.append("Tụt HA")
        
        if flags or ews >= 5:
            triage, color = "🔴 ĐỎ (CẤP CỨU KHẨN CẤP)", "#ef4444"
        elif ews >= 3 or chest_pain or pain >= 7:
            triage, color = "🟡 VÀNG (CẦP CỨU)", "#f59e0b"
        else:
            triage, color = "🟢 XANH (ỔN ĐỊNH)", "#10b981"

        # --- HIỂN THỊ KẾT QUẢ ---
        st.markdown(f"""
            <div class="triage-box" style="background-color: {color};">
                {triage}
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Early Warning Score (EWS)", ews)
        c2.metric("Điểm GCS", f"{gcs_total}/15")
        c3.metric("Shock Index", round(hr/sbp, 2) if sbp > 0 else 0)

        if flags:
            st.error(f"⚠️ **Dấu hiệu đe dọa:** {', '.join(flags)}")

        # Lưu vào log
        st.session_state["logs"].append({
            "Thời gian": datetime.now().strftime("%H:%M"),
            "Phân loại": triage,
            "EWS": ews,
            "GCS": gcs_total
        })

with tab2:
    if st.session_state["logs"]:
        df = pd.DataFrame(st.session_state["logs"])
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["Phân loại"].value_counts())
    else:
        st.info("Chưa có dữ liệu thống kê.")
