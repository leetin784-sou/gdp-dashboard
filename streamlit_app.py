import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
APP_TITLE = "TriageAI – Risk + Uncertainty"
APP_SUBTITLE = "Demo hỗ trợ phân luồng: định lượng rủi ro nguy kịch + độ không chắc chắn. Bác sĩ quyết định cuối."

st.set_page_config(page_title=APP_TITLE, layout="wide")

# =========================
# Data models
# =========================
@dataclass
class PatientInput:
    age: int
    hr: int
    sbp: int
    spo2: int
    rr: int
    temp: float
    avpu: str
    chest_pain: bool
    trauma: bool
    severe_dyspnea: bool

# =========================
# Helpers (math/ML demo)
# =========================
def sigmoid(x: float) -> float:
    # chống overflow nhẹ
    x = max(min(x, 40), -40)
    return 1.0 / (1.0 + math.exp(-x))

def avpu_index(avpu: str) -> int:
    mapping = {"A": 0, "V": 1, "P": 2, "U": 3}
    return mapping.get(avpu, 0)

def red_flags(p: PatientInput) -> list[str]:
    """Luật y khoa đơn giản để demo (có thể thay theo tài liệu triage bạn dùng)."""
    flags = []
    if p.spo2 < 90:
        flags.append("SpO₂ < 90%")
    if p.sbp < 90:
        flags.append("SBP < 90 mmHg")
    if avpu_index(p.avpu) >= 2:
        flags.append("Tri giác P/U")
    if p.severe_dyspnea:
        flags.append("Khó thở nặng")
    if p.hr >= 140:
        flags.append("HR ≥ 140")
    if p.rr >= 30:
        flags.append("RR ≥ 30")
    return flags

def risk_logistic_demo(p: PatientInput) -> float:
    """
    Demo risk model (KHÔNG phải model y tế thật).
    Trả về xác suất nguy kịch p in [0,1].
    """
    a = avpu_index(p.avpu)

    z = (
        -7.0
        + 0.015 * p.age
        + 0.020 * max(0, p.hr - 90)
        + 0.045 * max(0, 100 - p.sbp)
        + 0.095 * max(0, 95 - p.spo2)
        + 0.030 * max(0, p.rr - 18)
        + 0.50  * max(0, p.temp - 37.5)
        + 0.90  * a
        + 0.35  * (1 if p.chest_pain else 0)
        + 0.45  * (1 if p.trauma else 0)
    )
    return sigmoid(z)

def bootstrap_uncertainty(p: PatientInput, n: int = 35, seed: int = 42) -> tuple[float, float]:
    """
    Uncertainty demo: jitter đo đạc + lấy std của dự đoán.
    u càng cao => càng không chắc.
    """
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n):
        pj = PatientInput(
            age=int(np.clip(p.age + rng.normal(0, 1.5), 0, 120)),
            hr=int(np.clip(p.hr + rng.normal(0, 4.0), 30, 220)),
            sbp=int(np.clip(p.sbp + rng.normal(0, 4.0), 50, 220)),
            spo2=int(np.clip(p.spo2 + rng.normal(0, 1.0), 50, 100)),
            rr=int(np.clip(p.rr + rng.normal(0, 2.0), 5, 60)),
            temp=float(np.clip(p.temp + rng.normal(0, 0.15), 34.0, 42.0)),
            avpu=p.avpu,
            chest_pain=p.chest_pain,
            trauma=p.trauma,
            severe_dyspnea=p.severe_dyspnea,
        )
        preds.append(risk_logistic_demo(pj))

    preds = np.array(preds, dtype=float)
    return float(preds.mean()), float(preds.std(ddof=1))

def uncertainty_level(u: float) -> str:
    if u >= 0.20:
        return "CAO"
    if u >= 0.10:
        return "TRUNG BÌNH"
    return "THẤP"

def triage_from_risk(risk: float) -> str:
    if risk >= 0.70:
        return "🔴 ĐỎ"
    if risk >= 0.30:
        return "🟡 VÀNG"
    return "🟢 XANH"

def decision_message(risk: float, u: float, flags: list[str]) -> tuple[str, str]:
    """
    Human-in-the-loop:
    - Có red flag => ưu tiên Đỏ ngay
    - Không có => theo Risk + Uncertainty
    """
    if flags:
        return "🔴 ĐỎ (Red flags)", "Cảnh báo theo luật y khoa: " + "; ".join(flags)

    triage = triage_from_risk(risk)
    ul = uncertainty_level(u)

    if ul == "CAO":
        note = "⚠️ Uncertainty CAO: khuyến nghị bác sĩ đánh giá kỹ trước khi chốt."
    elif ul == "TRUNG BÌNH":
        note = "Uncertainty TRUNG BÌNH: nên kiểm tra thêm dấu hiệu/khai thác."
    else:
        note = "Uncertainty THẤP: mô hình khá tự tin."

    return triage, note

def top_reasons(p: PatientInput) -> list[str]:
    """
    “Giải thích” đơn giản theo rule-based để demo (không phải SHAP).
    """
    reasons = []
    if p.spo2 < 94: reasons.append("SpO₂ thấp")
    if p.sbp < 100: reasons.append("Huyết áp thấp")
    if p.hr > 110: reasons.append("Mạch nhanh")
    if p.rr > 22: reasons.append("Nhịp thở nhanh")
    if avpu_index(p.avpu) >= 1: reasons.append("Tri giác giảm")
    if p.chest_pain: reasons.append("Đau ngực")
    if p.trauma: reasons.append("Chấn thương")
    if p.severe_dyspnea: reasons.append("Khó thở nặng")
    return reasons[:5] if reasons else ["Không có yếu tố nổi bật"]

# =========================
# UI
# =========================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Cấu hình demo")
    n_boot = st.slider("Số lần bootstrap (tính Uncertainty)", 15, 80, 35, 5)
    st.markdown("---")
    st.caption("⚠️ Demo phục vụ thuyết trình/ý tưởng. Không dùng cho quyết định lâm sàng thật.")

tab1, tab2 = st.tabs(["🧾 Đánh giá ca", "📤 Xuất báo cáo"])

with tab1:
    colL, colR = st.columns([1, 1])

    with colL:
        st.subheader("Nhập dữ liệu ban đầu")
        age = st.number_input("Tuổi", 0, 120, 40)
        hr = st.number_input("Mạch (HR, bpm)", 30, 220, 90)
        sbp = st.number_input("Huyết áp tâm thu (SBP, mmHg)", 50, 220, 120)
        spo2 = st.number_input("SpO₂ (%)", 50, 100, 98)
        rr = st.number_input("Nhịp thở (RR, /phút)", 5, 60, 18)
        temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 37.0, 0.1)

    with colR:
        st.subheader("Triệu chứng / bối cảnh")
        avpu = st.selectbox("Tri giác (AVPU)", ["A", "V", "P", "U"], index=0, help="A: tỉnh, V: đáp ứng lời, P: đáp ứng đau, U: không đáp ứng")
        chest_pain = st.checkbox("Đau ngực")
        trauma = st.checkbox("Chấn thương")
        severe_dyspnea = st.checkbox("Khó thở nặng")

        st.markdown("### Kiểm tra nhanh")
        st.info("Nhập xong bấm **Đánh giá** để xem Risk + Uncertainty + gợi ý phân luồng.")

    p = PatientInput(
        age=int(age),
        hr=int(hr),
        sbp=int(sbp),
        spo2=int(spo2),
        rr=int(rr),
        temp=float(temp),
        avpu=str(avpu),
        chest_pain=bool(chest_pain),
        trauma=bool(trauma),
        severe_dyspnea=bool(severe_dyspnea),
    )

    st.markdown("---")

    if st.button("Đánh giá", type="primary", use_container_width=True):
        flags = red_flags(p)
        mean_risk, u = bootstrap_uncertainty(p, n=n_boot, seed=42)
        triage, note = decision_message(mean_risk, u, flags)
        reasons = top_reasons(p)

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score (P nguy kịch)", f"{mean_risk*100:.1f}%")
        c2.metric("Uncertainty (σ)", f"{u:.3f}")
        c3.metric("Mức tin cậy", uncertainty_level(u))

        if "🔴" in triage:
            st.error(f"**Gợi ý phân luồng:** {triage}\n\n{note}")
        elif "🟡" in triage:
            st.warning(f"**Gợi ý phân luồng:** {triage}\n\n{note}")
        else:
            st.success(f"**Gợi ý phân luồng:** {triage}\n\n{note}")

        st.markdown("### Giải thích (demo)")
        st.write("• " + "\n• ".join(reasons))

        # Lưu vào session để xuất báo cáo
        st.session_state["last_result"] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "age": p.age, "hr": p.hr, "sbp": p.sbp, "spo2": p.spo2, "rr": p.rr, "temp": p.temp,
            "avpu": p.avpu, "chest_pain": p.chest_pain, "trauma": p.trauma, "severe_dyspnea": p.severe_dyspnea,
            "risk": mean_risk, "uncertainty": u, "uncertainty_level": uncertainty_level(u),
            "triage": triage, "red_flags": "; ".join(flags) if flags else "",
            "reasons": "; ".join(reasons),
        }

with tab2:
    st.subheader("Xuất báo cáo ca (CSV)")
    last = st.session_state.get("last_result")
    if not last:
        st.warning("Chưa có kết quả nào. Vào tab **Đánh giá ca** rồi bấm **Đánh giá**.")
    else:
        df = pd.DataFrame([last])
        st.dataframe(df, use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tải báo cáo CSV",
            data=csv_bytes,
            file_name="triageai_case_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
