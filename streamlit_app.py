from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# METADATA
# =========================
APP_NAME = "General Emergency Triage AI"
APP_VERSION = "3.6 – General / Safety-first + Dept Routing"
MODEL_DESC = "Rule-based + Risk + Uncertainty + Human-in-the-loop + Department Recommendation"

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
def sigmoid(x: float) -> float:
    x = max(min(x, 40), -40)
    return 1.0 / (1.0 + math.exp(-x))

def avpu_idx(a: str) -> int:
    return {"A": 0, "V": 1, "P": 2, "U": 3}.get(a, 0)

# =========================
# INPUT VALIDATION
# =========================
def validate(p: Patient) -> Tuple[bool, List[str]]:
    hard, soft = [], []

    if not (0 <= p.age <= 120): hard.append("Tuổi ngoài phạm vi 0–120")
    if not (30 <= p.hr <= 220): hard.append("HR ngoài phạm vi 30–220")
    if not (50 <= p.sbp <= 250): hard.append("SBP ngoài phạm vi 50–250")
    if not (50 <= p.spo2 <= 100): hard.append("SpO₂ ngoài phạm vi 50–100")
    if not (5 <= p.rr <= 60): hard.append("RR ngoài phạm vi 5–60")
    if not (34.0 <= p.temp <= 42.0): hard.append("Nhiệt độ ngoài phạm vi 34–42°C")
    if p.avpu not in ["A", "V", "P", "U"]: hard.append("AVPU không hợp lệ")

    if p.spo2 < 88 and not p.dyspnea:
        soft.append("SpO₂ rất thấp nhưng chưa tick ‘Khó thở’ (kiểm tra lại).")

    return len(hard) == 0, hard + soft

# =========================
# RED FLAGS – HARD SAFETY
# =========================
def red_flags(p: Patient) -> List[str]:
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
def features(p: Patient) -> Dict[str, float]:
    return {
        "age": float(p.age),
        "hr": float(max(0, p.hr - 90)),
        "sbp": float(max(0, 100 - p.sbp)),
        "spo2": float(max(0, 95 - p.spo2)),
        "rr": float(max(0, p.rr - 18)),
        "temp": float(max(0, p.temp - 37.5)),
        "avpu": float(avpu_idx(p.avpu)),
        "chest_pain": float(int(p.chest_pain)),
        "trauma": float(int(p.trauma)),
        "dyspnea": float(int(p.dyspnea)),
        "altered_mental": float(int(p.altered_mental)),
    }

def ensemble_predict(p: Patient) -> Tuple[float, float]:
    base = {
        "b0": -7.0,
        "age": 0.010,
        "hr": 0.020,
        "sbp": 0.050,
        "spo2": 0.120,
        "rr": 0.030,
        "temp": 0.40,
        "avpu": 1.00,
        "chest_pain": 0.30,
        "trauma": 0.40,
        "dyspnea": 0.60,
        "altered_mental": 0.70,
    }

    rng = np.random.default_rng(42)
    probs = []
    x = features(p)

    for _ in range(17):
        z = base["b0"] + rng.normal(0, 0.30)
        for k, v in x.items():
            z += base[k] * (1 + rng.normal(0, 0.10)) * v
        probs.append(sigmoid(z))

    probs = np.array(probs, dtype=float)
    return float(probs.mean()), float(probs.std(ddof=1))

def uncertainty_level(u: float) -> str:
    if u >= 0.20: return "CAO"
    if u >= 0.10: return "TRUNG BÌNH"
    return "THẤP"

def triage_from_risk(r: float) -> str:
    if r >= 0.70: return "🔴 ĐỎ"
    if r >= 0.30: return "🟡 VÀNG"
    return "🟢 XANH"

# =========================
# DECISION + EXPLANATION
# =========================
def decision(r: float, u: float, flags: List[str]) -> Tuple[str, str]:
    if flags:
        return "🔴 ĐỎ (Luật an toàn)", "Kích hoạt red flags: " + "; ".join(flags)

    ul = uncertainty_level(u)
    if ul == "CAO":
        note = "Uncertainty CAO → không khuyến nghị mạnh; cần bác sĩ đánh giá."
    elif ul == "TRUNG BÌNH":
        note = "Uncertainty TRUNG BÌNH → nên đo lại vitals / bổ sung ngữ cảnh."
    else:
        note = "Uncertainty THẤP → mô hình tương đối chắc (bác sĩ quyết định cuối)."

    return triage_from_risk(r), note

# =========================
# NEW: RECOMMEND DEPARTMENT + REASONS
# =========================
def recommend_department(p: Patient, triage: str, flags: List[str]) -> Tuple[str, str]:
    """
    Safety-first routing:
    - Nếu ĐỎ hoặc có flags: ưu tiên Cấp cứu/Hồi sức trước, rồi định hướng chuyên khoa.
    - Nếu không: định hướng theo triệu chứng/vitals.
    """
    # 0) Pediatric quick rule (optional)
    is_peds = p.age < 16

    # 1) RED / flags -> resus first
    if flags or ("🔴" in triage):
        if is_peds:
            base = "Cấp cứu/Hồi sức (ưu tiên) → Nhi"
            reason = "Nguy kịch/Red flags + tuổi nhi."
            return base, reason

        if p.trauma:
            return "Cấp cứu/Hồi sức (ưu tiên) → Ngoại/Chấn thương", "Red flags/ĐỎ + chấn thương."
        if p.chest_pain:
            return "Cấp cứu/Hồi sức (ưu tiên) → Tim mạch", "Red flags/ĐỎ + đau ngực."
        if p.dyspnea or p.spo2 < 94:
            return "Cấp cứu/Hồi sức (ưu tiên) → Hô hấp", "Red flags/ĐỎ + khó thở/SpO₂ giảm."
        if p.altered_mental or avpu_idx(p.avpu) >= 2:
            return "Cấp cứu/Hồi sức (ưu tiên) → Thần kinh", "Red flags/ĐỎ + rối loạn tri giác."
        return "Cấp cứu/Hồi sức (ưu tiên)", "Red flags/ĐỎ: ưu tiên ổn định ABC trước."

    # 2) Non-red: department by symptom cluster
    if is_peds:
        return "Nhi (hoặc Cấp cứu Nhi)", "Tuổi < 16."

    if p.trauma:
        return "Ngoại/Chấn thương chỉnh hình", "Chấn thương là triệu chứng chính."
    if p.altered_mental or (p.onset == "Đột ngột" and p.progression == "Nặng dần"):
        return "Thần kinh", "Rối loạn tri giác / diễn tiến đáng ngại."
    if p.chest_pain:
        return "Tim mạch", "Đau ngực: cần ECG/men tim theo quy trình."
    if p.dyspnea or p.spo2 < 94:
        return "Hô hấp", "Khó thở/SpO₂ giảm."
    if p.temp >= 38.5 and (p.hr >= 110 or p.rr >= 22):
        return "Nội tổng quát / Nhiễm (tuỳ bệnh viện)", "Gợi ý nhiễm trùng: sốt + đáp ứng viêm."

    # 3) Default
    return "Cấp cứu tổng quát / Nội tổng quát", "Không có cụm triệu chứng nổi trội."

# =========================
# UI
# =========================
st.title(APP_NAME)
st.caption(f"{APP_VERSION} | {MODEL_DESC}")

tab1, tab2, tab3 = st.tabs(["🧾 Đánh giá", "🔍 Giải thích chuyên sâu", "📤 Logs/Export"])

# Ensure logs exists
st.session_state.setdefault("logs", [])
st.session_state.setdefault("last_case", None)

with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Vitals")
        age = st.number_input("Tuổi", 0, 120, 40)
        hr = st.number_input("HR", 30, 220, 90)
        sbp = st.number_input("SBP", 50, 250, 120)
        spo2 = st.number_input("SpO₂", 50, 100, 98)

    with c2:
        st.subheader("Vitals (cont.)")
        rr = st.number_input("RR", 5, 60, 18)
        temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 37.0, 0.1)
        avpu = st.selectbox("AVPU", ["A", "V", "P", "U"])

    with c3:
        st.subheader("Context")
        chest_pain = st.checkbox("Đau ngực")
        dyspnea = st.checkbox("Khó thở")
        trauma = st.checkbox("Chấn thương")
        altered_mental = st.checkbox("Rối loạn tri giác")

        onset = st.selectbox("Khởi phát", ["Đột ngột", "Từ từ"])
        progression = st.selectbox("Diễn tiến", ["Nặng dần", "Ổn định", "Giảm"])

    p = Patient(
        age=int(age), hr=int(hr), sbp=int(sbp), spo2=int(spo2),
        rr=int(rr), temp=float(temp), avpu=str(avpu),
        chest_pain=bool(chest_pain), dyspnea=bool(dyspnea), trauma=bool(trauma), altered_mental=bool(altered_mental),
        onset=str(onset), progression=str(progression)
    )

    ok, issues = validate(p)
    if issues:
        st.warning("Kiểm tra dữ liệu:\n- " + "\n- ".join(issues))

    run = st.button("Đánh giá", type="primary", disabled=not ok, use_container_width=True)

    if run:
        flags = red_flags(p)
        r, u = ensemble_predict(p)
        triage, note = decision(r, u, flags)

        dept, dept_reason = recommend_department(p, triage, flags)

        a, b, c = st.columns(3)
        a.metric("Risk (%)", f"{r*100:.1f}")
        b.metric("Uncertainty (σ)", f"{u:.3f}")
        c.metric("Độ không chắc", uncertainty_level(u))  # đổi tên cho khỏi hiểu nhầm

        if "🔴" in triage:
            st.error(f"**{triage}** — {note}")
        elif "🟡" in triage:
            st.warning(f"**{triage}** — {note}")
        else:
            st.success(f"**{triage}** — {note}")

        st.markdown("### 🏥 Đề xuất chuyển khoa")
        st.write(f"**{dept}**")
        st.caption(f"Lý do: {dept_reason}")

        # Save case once (no duplicate append)
        last_case = {
            **asdict(p),
            "risk": r,
            "uncertainty": u,
            "uncertainty_level": uncertainty_level(u),
            "triage": triage,
            "note": note,
            "department": dept,
            "department_reason": dept_reason,
            "red_flags": "; ".join(flags),
            "time": datetime.now().isoformat(timespec="seconds"),
            "app_version": APP_VERSION,
        }
        st.session_state["last_case"] = last_case
        st.session_state["logs"].append(last_case)

with tab2:
    st.subheader("Giải thích chuyên sâu (Why this decision?)")
    case = st.session_state.get("last_case")
    if not case:
        st.info("Chưa có ca nào được đánh giá. Vào tab **Đánh giá** và bấm **Đánh giá**.")
    else:
        st.markdown("### 1️⃣ Luật an toàn (Hard rules)")
        st.write("Nếu có **red flags** → ưu tiên **ĐỎ** ngay, không phụ thuộc AI.")
        st.write(f"Red flags: {case.get('red_flags') or 'Không'}")

        st.markdown("### 2️⃣ Risk score (tại sao ra % này?)")
        st.write("Risk được tính từ các dấu hiệu sinh tồn + context (đau ngực/khó thở/chấn thương/tri giác).")
        st.write(f"Risk: **{case['risk']*100:.1f}%**")

        st.markdown("### 3️⃣ Uncertainty (vì sao chắc/không chắc?)")
        st.write(
            "Uncertainty phản ánh mức **bất đồng** giữa nhiều mô hình (ensemble). "
            "Càng gần ngưỡng hoặc dữ liệu mâu thuẫn/thiếu context → uncertainty tăng."
        )
        st.write(f"Uncertainty σ: **{case['uncertainty']:.3f}** ({case['uncertainty_level']})")

        st.markdown("### 4️⃣ Human‑in‑the‑loop (AI làm gì, bác sĩ làm gì?)")
        st.write(
            "AI chỉ **đề xuất**. Khi **uncertainty cao**, hệ thống không áp đặt mà yêu cầu bác sĩ đánh giá."
        )

        st.markdown("### 5️⃣ Vì sao chuyển khoa này?")
        st.write(f"**{case['department']}**")
        st.caption(f"Lý do: {case['department_reason']}")

with tab3:
    st.subheader("Logs / Export (Audit trail)")
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("Chưa có log.")
    else:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, height=380)
        st.download_button(
            "Tải CSV",
            df.to_csv(index=False).encode("utf-8"),
            "triage_logs.csv",
            "text/csv",
            use_container_width=True
        )
