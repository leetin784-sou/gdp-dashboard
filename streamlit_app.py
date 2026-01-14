from __future__ import annotations
import math
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass, asdict
from datetime import datetime

# =========================
# CẤU HÌNH HỆ THỐNG & STYLE
# =========================
st.set_page_config(page_title="AI Triage Pro v4.0", layout="wide", page_icon="🚑")

# Tùy chỉnh CSS để giao diện giống phần mềm bệnh viện chuyên dụng
st.markdown("""
    <style>
    .main { background-color: #f0f2f5; }
    .stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 10px; }
    div[data-testid="stForm"] { background-color: white; border-radius: 15px; padding: 30px; border: none; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# =========================
# HÀM BỔ TRỢ LÂM SÀNG
# =========================
def calculate_ews(hr, rr, sbp, temp, spo2):
    """Tính điểm Early Warning Score đơn giản"""
    score = 0
    if hr > 110 or hr < 50: score += 2
    if rr > 24 or rr < 10: score += 2
    if sbp < 90 or sbp > 180: score += 2
    if temp > 38.5 or temp < 35.5: score += 1
    if spo2 < 94: score += 3
    return score

def get_gcs_desc(total):
    if total <= 8: return "Hôn mê sâu (Nặng)"
    if total <= 12: return "Tri giác u ám (Trung bình)"
    return "Tỉnh táo / Chấn thương nhẹ"

# =========================
# QUẢN LÝ DỮ LIỆU (STATE)
# =========================
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# =========================
# SIDEBAR - DASHBOARD TỔNG QUAN
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("Phòng Điều hành")
    
    df_logs = pd.DataFrame(st.session_state["logs"])
    
    if not df_logs.empty:
        st.metric("Tổng ca tiếp nhận", len(df_logs))
        red_cases = len(df_logs[df_logs['Phân loại'].str.contains("🔴")])
        st.metric("Ca Nguy kịch (Đỏ)", red_count := red_cases, delta=red_count, delta_color="inverse")
        
        st.divider()
        st.subheader("📊 Tỷ lệ phân loại")
        st.bar_chart(df_logs['Phân loại'].value_counts())
    
    if st.button("🔄 Làm mới toàn bộ hệ thống"):
        st.session_state["logs"] = []
        st.rerun()

# =========================
# GIAO DIỆN CHÍNH
# =========================
st.title("🚑 Hệ thống Phân loại Cấp cứu & Hỗ trợ Chẩn đoán AI")
st.caption(f"Phiên bản 4.0 Pro | Cập nhật: {datetime.now().strftime('%d/%m/%Y')}")

tab1, tab2, tab3 = st.tabs(["📝 Tiếp nhận Bệnh nhân", "📈 Phân tích Khoa", "⚙️ Cài đặt"])

with tab1:
    with st.form("triage_form_v4"):
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        
        with col1:
            st.markdown("### 🩸 Sinh hiệu (Vitals)")
            age = st.number_input("Tuổi", 0, 120, 35)
            hr = st.number_input("Nhịp tim (BPM)", 20, 250, 80)
            sbp = st.number_input("HA Tâm thu (mmHg)", 40, 250, 120)
            spo2 = st.slider("SpO₂ (%)", 70, 100, 98)
            rr = st.number_input("Nhịp thở (lần/phút)", 8, 50, 18)
            temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 36.6, 0.1)

        with col2:
            st.markdown("### 🧠 Thần kinh (GCS)")
            # Sử dụng Image tag để hướng dẫn chấm GCS
            
            gcs_e = st.selectbox("Mở mắt (Eye)", [4, 3, 2, 1], format_func=lambda x: f"{x} - {['Không đáp ứng', 'Kích thích đau', 'Lời nói', 'Tự nhiên'][x-1]}")
            gcs_v = st.selectbox("Lời nói (Verbal)", [5, 4, 3, 2, 1], format_func=lambda x: f"{x} - {['Không đáp ứng', 'Tiếng rên rỉ', 'Từ ngữ không phù hợp', 'Lú lẫn', 'Định hướng đúng'][x-1]}")
            gcs_m = st.selectbox("Vận động (Motor)", [6, 5, 4, 3, 2, 1], format_func=lambda x: f"{x} - {['Không đáp ứng', 'Duỗi cứng', 'Gấp cứng', 'Rút lui khi đau', 'Đáp ứng đúng kích thích đau', 'Theo lệnh'][x-1]}")
            
            total_gcs = gcs_e + gcs_v + gcs_m
            st.info(f"**Tổng điểm GCS: {total_gcs}/15** ({get_gcs_desc(total_gcs)})")

        with col3:
            st.markdown("### 🚩 Triệu chứng chính")
            pain_level = st.select_slider("Mức độ đau (VAS)", options=range(11), value=0)
            c1, c2 = st.columns(2)
            with c1:
                chest_pain = st.checkbox("Đau ngực")
                dyspnea = st.checkbox("Khó thở")
            with c2:
                trauma = st.checkbox("Chấn thương")
                altered_mental = st.checkbox("Lú lẫn")
            
            st.markdown("---")
            onset = st.selectbox("Khởi phát", ["Từ từ", "Cấp tính/Đột ngột"])

        submit = st.form_submit_button("XÁC NHẬN PHÂN LOẠI", type="primary", use_container_width=True)

    if submit:
        # --- LOGIC PHÂN LOẠI ---
        flags = []
        ews_score = calculate_ews(hr, rr, sbp, temp, spo2)
        si = round(hr / sbp, 2) if sbp > 0 else 0
        
        # Tiêu chuẩn Đỏ
        if total_gcs <= 8 or spo2 < 90 or sbp < 85 or ews_score >= 5:
            triage = "🔴 ĐỎ (NGUY KỊCH)"
            color_hex = "#FF4B4B"
            advice = "Chuyển ngay vào phòng Hồi sức (Resus). Thiết lập đường truyền, hỗ trợ hô hấp."
        # Tiêu chuẩn Vàng
        elif ews_score >= 3 or si > 0.9 or chest_pain or pain_level >= 7:
            triage = "🟡 VÀNG (CẤP CỨU)"
            color_hex = "#FFA500"
            advice = "Ưu tiên thăm khám trong vòng 15-30 phút. Làm ECG/Xét nghiệm tại giường."
        # Tiêu chuẩn Xanh
        else:
            triage = "🟢 XANH (ÍT CẤP THIẾT)"
            color_hex = "#28A745"
            advice = "Bệnh nhân ổn định. Chuyển khu vực chờ khám nội khoa tổng quát."

        # --- HIỂN THỊ KẾT QUẢ ---
        st.markdown(f"<div style='background-color:{color_hex}; padding:20px; border-radius:10px; text-align:center; color:white;'><h1>{triage}</h1></div>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Điểm EWS", ews_score, delta="Nguy cơ" if ews_score > 3 else "An toàn", delta_color="inverse")
        m2.metric("Chỉ số Sốc (SI)", si)
        m3.metric("Điểm GCS", f"{total_gcs}/15")
        m4.metric("Đau (VAS)", f"{pain_level}/10")

        st.success(f"**Hướng xử trí:** {advice}")
        
        # SBAR Copy-paste
        sbar_text = f"SBAR REPORT: BN {age}T | GCS: {total_gcs} | HA: {sbp}mmHg | SpO2: {spo2}% | Triage: {triage}."
        st.text_area("Bản tóm tắt chuyên môn (SBAR):", sbar_text)

        # Lưu log
        st.session_state["logs"].append({
            "Thời gian": datetime.now().strftime("%H:%M:%S"),
            "Tuổi": age,
            "Phân loại": triage,
            "EWS": ews_score,
            "GCS": total_gcs,
            "HA/Mạch": f"{sbp}/{hr}"
        })

with tab2:
    if st.session_state["logs"]:
        st.subheader("📈 Phân tích lưu lượng bệnh nhân")
        df_analysis = pd.DataFrame(st.session_state["logs"])
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.write("Mức độ rủi ro (EWS) theo thời gian")
            st.line_chart(df_analysis.set_index("Thời gian")["EWS"])
        with col_chart2:
            st.write("Cơ cấu bệnh nhân theo phân loại")
            st.bar_chart(df_analysis["Phân loại"].value_counts())
            
        st.subheader("📋 Nhật ký chi tiết")
        st.dataframe(df_analysis, use_container_width=True)
    else:
        st.info("Chưa có dữ liệu để hiển thị biểu đồ.")

with tab3:
    st.header("Cài đặt hệ thống")
    st.write("Cấu hình các ngưỡng cảnh báo (Sắp ra mắt...)")
    st.download_button("Xuất dữ liệu CSV", pd.DataFrame(st.session_state["logs"]).to_csv(), "hospital_logs.csv")
