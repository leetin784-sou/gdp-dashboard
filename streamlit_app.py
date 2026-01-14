from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# METADATA
# =========================
APP_NAME = "General Emergency Triage AI"
APP_VERSION = "2.1 – General / Safety-first"
MODEL_DESC = "Rule-based + Risk + Uncertainty + Human-in-the-loop"

st.set_page_config(page_title=APP_NAME, layout="wide")

# =========================
# DATA MODELS
# =========================
@dataclass
class Patient:
    age: int
    hr: int
    sbp: int
    spo2: int
    rr: int
    temp: float
    avpu: str

    chest_pain: bool
    dyspnea: bool
    trauma: bool
    altered_mental: bool

    onset: str
    progression: str

# =========================
# UTILITIES
# =========================
def sigmoid(x):
    x = max(min(x, 40), -40)
    return 1 / (1 + math.exp(-x))

def avpu_idx(a):
    return {"A": 0, "V": 1, "P": 2, "U": 3}.get(a, 0)

# =========================
# INPUT VALIDATION
# =========================
def validate(p: Patient):
    hard, soft = [], []

    if not (0 <= p.age <= 120): hard.append("Tuổi không hợp lệ")
    if not (30 <= p.hr <= 220): hard.append("HR ngoài phạm vi")
    if not (50 <= p.sbp <= 250): hard.append("SBP ngoài phạm vi")
    if not (50 <= p.spo2 <= 100): hard.append("SpO₂ ngoài phạm vi")
    if not (5 <= p.rr <= 60): hard.append("RR ngoài phạm vi")
    if not (34 <= p.temp <= 42): hard.append("Nhiệt độ ngoài phạm vi")

    if p.spo2 < 88 and not p.dyspnea:
        soft.append("SpO₂ thấp nhưng chưa ghi nhận khó thở")

    return len(hard) == 0, hard + soft

# =========================
# RED FLAGS – HARD SAFETY
# =========================
def red_flags(p: Patient):
    flags = []
    if p.spo2 < 90: flags.append("SpO₂ < 90%")
    if p.sbp < 90: flags.append("SBP < 90 mmHg")
    if avpu_idx(p.avpu) >= 2: flags.append("Tri giác giảm (AVPU P/U)")
    if p.dyspnea: flags.append("Khó thở rõ")
    if p.altered_mental: flags.append("Rối loạn tri giác")
    if p.hr >= 140: flags.append("HR ≥ 140")
    if p.rr >= 30: flags.append("RR ≥ 30")
    return flags

# =========================
# RISK MODEL (ENSEMBLE – NO TRAINING)
# =========================
def features(p: Patient):
    return {
        "age": p.age,
        "hr": max(0, p.hr - 90),
        "sbp": max(0, 100 - p.sbp),
        "spo2": max(0, 95 - p.spo2),
        "rr": max(0, p.rr - 18),
        "temp": max(0, p.temp - 37.5),
        "avpu": avpu_idx(p.avpu),
        "chest_pain": int(p.chest_pain),
        "trauma": int(p.trauma),
        "dyspnea": int(p.dyspnea),
    }

def ensemble_predict(p: Patient):
    base = {
        "b0": -7.0,
        "age": 0.01,
        "hr": 0.02,
        "sbp": 0.05,
        "spo2": 0.12,
        "rr": 0.03,
        "temp": 0.4,
        "avpu": 1.0,
        "chest_pain": 0.3,
        "trauma": 0.4,
        "dyspnea": 0.6,
    }

    rng = np.random.default_rng(42)
    probs = []

    for _ in range(15):
        z = base["b0"] + rng.normal(0, 0.3)
        for k, v in features(p).items():
            z += base[k] * (1 + rng.normal(0, 0.1)) * v
        probs.append(sigmoid(z))

    probs = np.array(probs)
    return probs.mean(), probs.std(ddof=1)

def uncertainty_level(u):
    if u >= 0.20: return "CAO"
    if u >= 0.10: return "TRUNG BÌNH"
    return "THẤP"

def triage_from_risk(r):
    if r >= 0.70: return "🔴 ĐỎ"
    if r >= 0.30: return "🟡 VÀNG"
    return "🟢 XANH"

# =========================
# DECISION + EXPLANATION
# =========================
def decision(p, r, u, flags):
    if flags:
        return "🔴 ĐỎ (Luật an toàn)", "Kích hoạt red flags: " + "; ".join(flags)

    note = ""
    if uncertainty_level(u) == "CAO":
        note = "Độ không chắc chắn cao → cần bác sĩ đánh giá"
    elif uncertainty_level(u) == "TRUNG BÌNH":
        note = "Nên đo lại vitals / bổ sung thông tin"
    else:
        note = "AI tương đối chắc chắn"

    return triage_from_risk(r), note

# =========================
# UI
# =========================
st.title(APP_NAME)
st.caption(f"{APP_VERSION} | {MODEL_DESC}")

tab1, tab2, tab3 = st.tabs(["🧾 Đánh giá", "🔍 Giải thích chuyên sâu", "📤 Logs"])

with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Tuổi", 0, 120, 40)
        hr = st.number_input("HR", 30, 220, 90)
        sbp = st.number_input("SBP", 50, 250, 120)
        spo2 = st.number_input("SpO₂", 50, 100, 98)

    with c2:
        rr = st.number_input("RR", 5, 60, 18)
        temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 37.0, 0.1)
        avpu = st.selectbox("AVPU", ["A", "V", "P", "U"])

    with c3:
        chest_pain = st.checkbox("Đau ngực")
        dyspnea = st.checkbox("Khó thở")
        trauma = st.checkbox("Chấn thương")
        altered_mental = st.checkbox("Rối loạn tri giác")

        onset = st.selectbox("Khởi phát", ["Đột ngột", "Từ từ"])
        progression = st.selectbox("Diễn tiến", ["Nặng dần", "Ổn định", "Giảm"])

    p = Patient(age, hr, sbp, spo2, rr, temp, avpu,
                chest_pain, dyspnea, trauma, altered_mental,
                onset, progression)

    ok, issues = validate(p)
    if issues:
        st.warning("Kiểm tra dữ liệu:\n- " + "\n- ".join(issues))

    if st.button("Đánh giá", disabled=not ok):
        flags = red_flags(p)
        r, u = ensemble_predict(p)
        triage, note = decision(p, r, u, flags)

        a, b, c = st.columns(3)
        a.metric("Risk (%)", f"{r*100:.1f}")
        b.metric("Uncertainty (σ)", f"{u:.3f}")
        c.metric("Mức tin cậy", uncertainty_level(u))

        if "🔴" in triage:
            st.error(triage + " – " + note)
        elif "🟡" in triage:
            st.warning(triage + " – " + note)
        else:
            st.success(triage + " – " + note)

        st.session_state["last_case"] = {
            **asdict(p),
            "risk": r,
            "uncertainty": u,
            "triage": triage,
            "note": note,
            "time": datetime.now().isoformat(timespec="seconds")
        }

with tab2:
    st.subheader("Giải thích chuyên sâu (Why this decision?)")
    case = st.session_state.get("last_case")
    if not case:
        st.info("Chưa có ca nào được đánh giá.")
    else:
        st.markdown("### 1️⃣ Luật an toàn")
        st.write("Nếu có red flags → ưu tiên ĐỎ, không phụ thuộc AI.")

        st.markdown("### 2️⃣ Risk score")
        st.write("Risk phản ánh xác suất nguy kịch dựa trên nhiều yếu tố sinh tồn.")

        st.markdown("### 3️⃣ Uncertainty")
        st.write(
            "Uncertainty cao khi dữ liệu sát ngưỡng hoặc mâu thuẫn. "
            "Hệ thống chủ động yêu cầu bác sĩ đánh giá để tránh quyết định sai."
        )

        st.markdown("### 4️⃣ Human‑in‑the‑loop")
        st.write(
            "AI không ra quyết định cuối. "
            "Khi không chắc, hệ thống chuyển quyền cho bác sĩ."
        )

with tab3:
    logs = st.session_state.get("logs", [])
    if "last_case" in st.session_state:
        st.session_state.setdefault("logs", []).append(st.session_state["last_case"])

    if st.session_state.get("logs"):
        df = pd.DataFrame(st.session_state["logs"])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Tải CSV",
            df.to_csv(index=False).encode("utf-8"),
            "triage_logs.csv",
            "text/csv"
        )
    else:
        st.info("Chưa có log.")
