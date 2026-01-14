import math
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass, asdict
from datetime import datetime

# =========================
# CONFIG & STYLE
# =========================
st.set_page_config(page_title="Smart Triage AI Pro", layout="wide", page_icon="🚑")

st.markdown("""
<style>
.main { background-color: #0b1220; }
.block-container { padding-top: 1.2rem; }

/* Fix metric visibility on dark background */
[data-testid="stMetric"] {
    background-color: #111827 !important;
    color: #F9FAFB !important;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #1F2937;
}
[data-testid="stMetricLabel"] {
    color: #9CA3AF !important;
    font-size: 0.9rem;
}
[data-testid="stMetricValue"] {
    color: #F9FAFB !important;
    font-size: 1.6rem;
    font-weight: 700;
}

/* Header */
.triage-header { text-align: center; padding: 18px; border-radius: 12px; color: white; margin: 8px 0 18px 0; }
.small-note { color: #9CA3AF; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v6.0 – High-Trust (EWS+GCS+SI + AI Risk/Unc + Context + Dept + Protocol + Audit)"
MODEL_NOTE = "Safety-first: Red flags > Clinical scores > AI Risk+Uncertainty (Human-in-the-loop)."

# =========================
# DATA MODEL
# =========================
@dataclass
class Patient:
    # Vitals
    age: int
    hr: int
    sbp: int
    spo2: int
    rr: int
    temp: float

    # GCS
    gcs_e: int
    gcs_v: int
    gcs_m: int

    # Symptoms
    chest_pain: bool
    dyspnea: bool
    trauma: bool
    pain_level: int

    # Course
    onset: str
    progression: str

    # Extra context (new)
    fast_stroke: bool
    bleeding: bool
    abdominal_pain: bool
    pregnancy: bool
    infection_suspected: bool
    anaphylaxis: bool
    poisoning_overdose: bool

# =========================
# CLINICAL UTILITIES
# =========================
def calculate_shock_index(hr, sbp):
    return round(hr / sbp, 2) if sbp > 0 else 0.0

def calculate_ews(hr, rr, sbp, temp, spo2):
    score = 0
    if hr > 110 or hr < 50: score += 2
    if rr > 24 or rr < 10: score += 2
    if sbp < 90 or sbp > 180: score += 2
    if temp > 38.5 or temp < 35.5: score += 1
    if spo2 < 94: score += 3
    return score

def gcs_total(p: Patient) -> int:
    return p.gcs_e + p.gcs_v + p.gcs_m

# =========================
# VALIDATION
# =========================
def validate_inputs(p: Patient):
    hard, soft = [], []
    if not (0 <= p.age <= 120): hard.append("Tuổi ngoài phạm vi 0–120.")
    if not (20 <= p.hr <= 250): hard.append("HR ngoài phạm vi 20–250.")
    if not (40 <= p.sbp <= 250): hard.append("SBP ngoài phạm vi 40–250.")
    if not (50 <= p.spo2 <= 100): hard.append("SpO₂ ngoài phạm vi 50–100%.")
    if not (5 <= p.rr <= 60): hard.append("RR ngoài phạm vi 5–60.")
    if not (34.0 <= p.temp <= 42.0): hard.append("Nhiệt độ ngoài phạm vi 34–42°C.")
    g = gcs_total(p)
    if not (3 <= g <= 15): hard.append("GCS không hợp lệ.")

    if p.spo2 < 88 and not p.dyspnea:
        soft.append("SpO₂ rất thấp nhưng chưa tick 'Khó thở' (kiểm tra lại).")

    if p.pregnancy and p.age < 10:
        soft.append("Mang thai + tuổi rất nhỏ (kiểm tra lại).")

    return (len(hard) == 0), hard + soft

# =========================
# RED FLAGS (HARD SAFETY)
# =========================
def red_flags(p: Patient, si: float, ews: int):
    g = gcs_total(p)
    flags = []

    if p.anaphylaxis:
        flags.append("Sốc phản vệ nghi ngờ")
    if g <= 8:
        flags.append("Hôn mê nặng (GCS ≤ 8)")
    if p.fast_stroke:
        flags.append("FAST dương tính (nghi đột quỵ)")
    if p.spo2 < 90:
        flags.append("Suy hô hấp nặng (SpO₂ < 90%)")
    if p.sbp < 90:
        flags.append("Sốc / tụt huyết áp (SBP < 90)")
    if si > 1.0:
        flags.append(f"Shock Index nguy hiểm ({si})")
    if p.rr >= 30:
        flags.append("Thở nhanh nặng (RR ≥ 30)")
    if p.hr >= 140:
        flags.append("Mạch nhanh nặng (HR ≥ 140)")
    if p.bleeding and (p.sbp < 100 or p.hr > 110):
        flags.append("Chảy máu + dấu hiệu huyết động xấu")
    if p.poisoning_overdose and g <= 12:
        flags.append("Nghi ngộ độc + giảm tri giác")
    if ews >= 7:
        flags.append("EWS rất cao (≥ 7)")

    return flags

# =========================
# AI RISK + UNCERTAINTY (NO TRAIN)
# =========================
def sigmoid(x: float) -> float:
    x = max(min(x, 40), -40)
    return 1.0 / (1.0 + math.exp(-x))

FEATURE_LABELS = {
    "spo2_drop": "SpO₂ thấp",
    "sbp_drop": "Huyết áp thấp",
    "hr_excess": "Mạch nhanh",
    "rr_excess": "Thở nhanh",
    "gcs_drop": "Tri giác giảm (GCS)",
    "temp_excess": "Sốt",
    "chest_pain": "Đau ngực",
    "dyspnea": "Khó thở",
    "trauma": "Chấn thương",
    "pain_hi": "Đau nhiều (VAS ≥ 7)",
    "onset_sudden": "Khởi phát đột ngột",
    "worsening": "Nặng dần",
    "fast_stroke": "FAST (+)",
    "bleeding": "Chảy máu",
    "abdominal_pain": "Đau bụng cấp",
    "pregnancy": "Mang thai",
    "infection": "Nghi nhiễm trùng",
    "anaphylaxis": "Sốc phản vệ",
    "poisoning": "Ngộ độc/quá liều",
    "age": "Tuổi",
}

def features(p: Patient):
    g = gcs_total(p)
    return {
        "age": float(p.age),
        "hr_excess": float(max(0, p.hr - 90)),
        "sbp_drop": float(max(0, 100 - p.sbp)),
        "spo2_drop": float(max(0, 95 - p.spo2)),
        "rr_excess": float(max(0, p.rr - 18)),
        "temp_excess": float(max(0, p.temp - 37.5)),
        "gcs_drop": float(max(0, 15 - g)),

        "chest_pain": float(int(p.chest_pain)),
        "dyspnea": float(int(p.dyspnea)),
        "trauma": float(int(p.trauma)),
        "pain_hi": float(int(p.pain_level >= 7)),

        "onset_sudden": float(int(p.onset == "Đột ngột")),
        "worsening": float(int(p.progression == "Nặng dần")),

        # extra context
        "fast_stroke": float(int(p.fast_stroke)),
        "bleeding": float(int(p.bleeding)),
        "abdominal_pain": float(int(p.abdominal_pain)),
        "pregnancy": float(int(p.pregnancy)),
        "infection": float(int(p.infection_suspected)),
        "anaphylaxis": float(int(p.anaphylaxis)),
        "poisoning": float(int(p.poisoning_overdose)),
    }

def ensemble_predict_with_explain(p: Patient):
    base = {
        "b0": -7.2,
        "age": 0.010,
        "hr_excess": 0.020,
        "sbp_drop": 0.050,
        "spo2_drop": 0.120,
        "rr_excess": 0.030,
        "temp_excess": 0.40,
        "gcs_drop": 0.55,

        "chest_pain": 0.25,
        "dyspnea": 0.55,
        "trauma": 0.35,
        "pain_hi": 0.15,

        "onset_sudden": 0.12,
        "worsening": 0.18,

        "fast_stroke": 0.80,
        "bleeding": 0.60,
        "abdominal_pain": 0.25,
        "pregnancy": 0.30,
        "infection": 0.45,
        "anaphylaxis": 1.20,
        "poisoning": 0.55,
    }

    x = features(p)
    rng = np.random.default_rng(42)
    preds = []
    for _ in range(25):
        z = base["b0"] + rng.normal(0, 0.30)
        for k, v in x.items():
            z += base[k] * (1 + rng.normal(0, 0.10)) * v
        preds.append(sigmoid(z))

    arr = np.array(preds, dtype=float)
    mean_r = float(arr.mean())
    std_u = float(arr.std(ddof=1))

    contrib = {k: float(base[k] * v) for k, v in x.items()}
    contrib_sorted = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return mean_r, std_u, contrib_sorted, preds

def uncertainty_level(u: float) -> str:
    if u >= 0.20: return "CAO"
    if u >= 0.10: return "TRUNG BÌNH"
    return "THẤP"

def triage_from_risk(r: float) -> str:
    if r >= 0.70: return "🔴 ĐỎ"
    if r >= 0.30: return "🟡 VÀNG"
    return "🟢 XANH"

# =========================
# DECISION POLICY
# =========================
def triage_decision(flags: list, ews: int, risk: float, u: float, p: Patient):
    if flags:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", "Luật an toàn kích hoạt: " + ", ".join(flags)

    # Protocol-friendly (EWS)
    if ews >= 5:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", f"EWS cao (≥5): {ews}. Ưu tiên đánh giá ngay."
    if ews >= 3 or p.chest_pain or p.pain_level >= 7 or p.fast_stroke or p.anaphylaxis or p.bleeding:
        note = f"Ưu tiên theo triệu chứng/điểm: EWS={ews}."
        if uncertainty_level(u) == "CAO":
            note += " Uncertainty CAO → cần bác sĩ xác nhận/đo lại."
        return "🟡 VÀNG (ƯU TIÊN)", "#FFA500", note

    # AI assist
    base = triage_from_risk(risk)
    if base.startswith("🔴"):
        if uncertainty_level(u) == "CAO":
            return "🟡 VÀNG (REVIEW)", "#FFA500", "Risk cao nhưng Uncertainty CAO → cần bác sĩ review."
        return "🔴 ĐỎ (CẢNH BÁO)", "#FF4B4B", "Risk cao & Uncertainty thấp → cảnh báo mạnh."
    if base.startswith("🟡"):
        if uncertainty_level(u) == "CAO":
            return "🟡 VÀNG (REVIEW)", "#FFA500", "Vùng xám + Uncertainty CAO → đo lại vitals/bổ sung ngữ cảnh."
        return "🟡 VÀNG (ƯU TIÊN)", "#FFA500", "Risk trung bình → theo dõi sát/khám ưu tiên."
    return "🟢 XANH (ỔN ĐỊNH)", "#28A745", "Risk thấp → ít nguy kịch (bác sĩ quyết định cuối)."

# =========================
# DEPARTMENT RECOMMENDATION
# =========================
def recommend_department(p: Patient, triage: str, flags: list):
    is_peds = p.age < 16
    g = gcs_total(p)

    # RED/flags: stabilize first
    if flags or ("🔴" in triage):
        if is_peds:
            return "Cấp cứu/Hồi sức (ưu tiên) → Nhi", "Nguy kịch + tuổi nhi."
        if p.anaphylaxis:
            return "Cấp cứu/Hồi sức (ưu tiên)", "Sốc phản vệ: ưu tiên ABC + xử trí phản vệ."
        if p.fast_stroke or g <= 12:
            return "Cấp cứu/Hồi sức (ưu tiên) → Thần kinh", "Giảm tri giác/FAST (+) → định hướng thần kinh."
        if p.bleeding:
            return "Cấp cứu/Hồi sức (ưu tiên) → Ngoại / Tiêu hoá", "Chảy máu: ưu tiên hồi sức, định hướng ngoại/tiêu hoá."
        if p.trauma:
            return "Cấp cứu/Hồi sức (ưu tiên) → Ngoại/Chấn thương", "Nguy kịch + chấn thương."
        if p.chest_pain:
            return "Cấp cứu/Hồi sức (ưu tiên) → Tim mạch", "Nguy kịch + đau ngực."
        if p.dyspnea or p.spo2 < 94:
            return "Cấp cứu/Hồi sức (ưu tiên) → Hô hấp", "Nguy kịch + khó thở/SpO₂ giảm."
        if p.poisoning_overdose:
            return "Cấp cứu/Hồi sức (ưu tiên) → Chống độc / Nội", "Ngộ độc/quá liều nghi ngờ."
        if p.pregnancy:
            return "Cấp cứu/Hồi sức (ưu tiên) → Sản", "Nguy kịch + thai kỳ."
        return "Cấp cứu/Hồi sức (ưu tiên)", "Ưu tiên ổn định ABC trước, sau đó phân khoa."

    # Non-red
    if is_peds:
        return "Nhi (hoặc Cấp cứu Nhi)", "Tuổi < 16."
    if p.pregnancy:
        return "Sản", "Thai kỳ."
    if p.fast_stroke or g <= 13:
        return "Thần kinh", "Nghi đột quỵ/tri giác giảm."
    if p.trauma:
        return "Ngoại/Chấn thương chỉnh hình", "Chấn thương."
    if p.bleeding:
        return "Tiêu hoá / Ngoại", "Chảy máu (GI/ngoài)."
    if p.abdominal_pain:
        return "Tiêu hoá / Ngoại", "Đau bụng cấp."
    if p.chest_pain:
        return "Tim mạch", "Đau ngực → ECG/men tim."
    if p.dyspnea or p.spo2 < 94:
        return "Hô hấp", "Khó thở/SpO₂ giảm."
    if p.infection_suspected:
        return "Nhiễm / Nội tổng quát", "Nghi nhiễm trùng."
    if p.poisoning_overdose:
        return "Chống độc / Nội", "Ngộ độc/quá liều."
    return "Cấp cứu tổng quát / Nội tổng quát", "Không có cụm triệu chứng nổi trội."

# =========================
# PROTOCOL ACTIONS (doctor-facing)
# =========================
def protocol_actions(dept: str, triage: str, p: Patient):
    actions = []
    # General by triage
    if "🔴" in triage:
        actions += [
            "Ưu tiên ABC: đường thở – hô hấp – tuần hoàn.",
            "Monitor, đo lại sinh hiệu sớm, đường truyền, chuẩn bị hồi sức.",
            "Bác sĩ đánh giá ngay."
        ]
    elif "🟡" in triage:
        actions += [
            "Khám ưu tiên, theo dõi sát, đo lại sinh hiệu.",
            "Làm cận lâm sàng theo triệu chứng."
        ]
    else:
        actions += [
            "Theo dõi cơ bản, tư vấn, hẹn tái khám nếu nặng lên."
        ]

    # Dept-specific
    if "Tim mạch" in dept:
        actions += ["ECG sớm (≤10 phút nếu nghi ACS).", "Xét nghiệm men tim theo protocol.", "Theo dõi đau ngực & dấu hiệu thiếu máu cơ tim."]
    if "Hô hấp" in dept:
        actions += ["Đánh giá đường thở/oxy, cân nhắc khí máu.", "X-quang phổi nếu phù hợp.", "Khí dung/thuốc theo tình huống lâm sàng."]
    if "Thần kinh" in dept:
        actions += ["Đánh giá FAST/GCS, kiểm tra đường huyết.", "Cân nhắc CT/đánh giá đột quỵ theo quy trình.", "Theo dõi tri giác liên tục."]
    if "Ngoại" in dept or "Chấn thương" in dept:
        actions += ["Kiểm soát chảy máu, bất động nếu chấn thương.", "Đánh giá tổn thương theo ABCDE.", "Cân nhắc siêu âm FAST trauma nếu phù hợp."]
    if "Tiêu hoá" in dept:
        actions += ["Đánh giá đau bụng/xuất huyết tiêu hoá.", "Theo dõi huyết động, xét nghiệm theo chỉ định.", "Cân nhắc hội chẩn nếu nặng."]
    if "Nhiễm" in dept:
        actions += ["Đánh giá sepsis: dấu hiệu nhiễm + huyết động.", "Xét nghiệm/cấy theo protocol nếu nghi nặng.", "Kháng sinh sớm nếu có chỉ định."]
    if "Sản" in dept:
        actions += ["Đánh giá thai kỳ/ra huyết/đau bụng.", "Theo dõi mẹ và thai (tuỳ điều kiện).", "Hội chẩn sản khi cần."]
    if "Chống độc" in dept:
        actions += ["Xác định chất/nghi ngờ quá liều.", "Theo dõi tri giác/hô hấp, cân nhắc giải độc.", "Liên hệ chống độc nếu cần."]

    # Context boosters
    if p.anaphylaxis:
        actions += ["Phác đồ phản vệ theo quy định (ưu tiên)."]
    if p.fast_stroke:
        actions += ["Kích hoạt đường dây đột quỵ (nếu có)."]
    if p.bleeding:
        actions += ["Đánh giá nguồn chảy máu, cân nhắc truyền dịch/máu theo chỉ định."]

    # Remove duplicates while preserving order
    seen = set()
    final = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            final.append(a)
    return final

# =========================
# EXPLANATION HELPERS
# =========================
def top_reasons(contrib_sorted, k=7):
    out = []
    for feat, val in list(contrib_sorted.items())[:k]:
        if abs(val) < 0.05:
            continue
        out.append(FEATURE_LABELS.get(feat, feat))
    return out if out else ["Không có yếu tố nổi bật"]

def decision_support_reasons(p: Patient, triage: str, flags: list, ews: int, si: float, risk: float, u: float, dept: str):
    g = gcs_total(p)
    reasons = []

    # Hard rules first
    if flags:
        reasons.append(f"❗ **Luật an toàn kích hoạt:** {', '.join(flags)}")
        if g <= 8:
            reasons.append("🧠 **GCS ≤ 8** → ưu tiên cấp cứu dù sinh hiệu khác có thể bình thường.")
        if p.fast_stroke:
            reasons.append("⚡ **FAST (+)** → nghi đột quỵ, cần xử trí theo quy trình.")
        if p.anaphylaxis:
            reasons.append("🚨 **Nghi phản vệ** → ưu tiên ABC + phác đồ phản vệ.")
    else:
        reasons.append("✅ Không kích hoạt red flags bắt buộc.")

    # Clinical scores
    reasons.append(f"📊 **EWS = {ews}**, Shock Index = **{si}**, GCS = **{g}/15**")

    # Course/context
    if p.onset == "Đột ngột":
        reasons.append("⚡ **Khởi phát đột ngột** → gợi ý biến cố cấp.")
    if p.progression == "Nặng dần":
        reasons.append("📈 **Nặng dần** → nguy cơ xấu nếu trì hoãn xử trí.")
    if p.bleeding:
        reasons.append("🩸 **Có chảy máu** → cần đánh giá huyết động & nguồn chảy máu.")
    if p.abdominal_pain:
        reasons.append("腹 **Đau bụng cấp** → định hướng tiêu hoá/ngoại.")
    if p.pregnancy:
        reasons.append("🤰 **Thai kỳ** → ưu tiên Sản khi phù hợp.")
    if p.infection_suspected:
        reasons.append("🦠 **Nghi nhiễm trùng** → cân nhắc sepsis theo protocol.")
    if p.poisoning_overdose:
        reasons.append("☠️ **Nghi ngộ độc/quá liều** → theo dõi tri giác/hô hấp.")

    # AI assist
    reasons.append(f"🤖 **AI Risk = {risk*100:.1f}%** (chỉ hỗ trợ, không override luật/protocol).")
    ul = uncertainty_level(u)
    if ul == "THẤP":
        reasons.append("📉 **Uncertainty thấp** → các mô hình đồng thuận cao.")
    elif ul == "TRUNG BÌNH":
        reasons.append("⚠️ **Uncertainty trung bình** → nên đo lại vitals/bổ sung ngữ cảnh.")
    else:
        reasons.append("🟠 **Uncertainty cao** → cần bác sĩ đánh giá trước khi quyết định mạnh.")

    # Dept routing
    reasons.append(f"🏥 **Định hướng khoa:** {dept}")

    return reasons

# =========================
# APP UI
# =========================
st.title("🏥 Smart Triage AI Pro – Hệ thống phân loại cấp cứu tổng quát")
st.caption(f"{APP_VERSION} | {MODEL_NOTE}")

if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "last_case" not in st.session_state:
    st.session_state["last_case"] = None

tab1, tab2, tab3 = st.tabs(["📝 Tiếp nhận", "📊 Dashboard", "📑 Nhật ký / Export + Explain"])

# -------------------------
# TAB 1: INTAKE
# -------------------------
with tab1:
    with st.form("triage_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🩸 Sinh hiệu")
            age = st.number_input("Tuổi", 0, 120, 35)
            hr = st.number_input("Nhịp tim (BPM)", 20, 250, 80)
            sbp = st.number_input("Huyết áp tâm thu (mmHg)", 40, 250, 120)
            spo2 = st.slider("SpO₂ (%)", 50, 100, 98)
            rr = st.number_input("Nhịp thở (/phút)", 5, 60, 18)
            temp = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 36.6, 0.1)

        with col2:
            st.subheader("🧠 Thần kinh (GCS)")
            e = st.selectbox("Mở mắt (E)", [4, 3, 2, 1], format_func=lambda x: f"{x} điểm")
            v = st.selectbox("Lời nói (V)", [5, 4, 3, 2, 1], format_func=lambda x: f"{x} điểm")
            m = st.selectbox("Vận động (M)", [6, 5, 4, 3, 2, 1], format_func=lambda x: f"{x} điểm")
            onset = st.selectbox("Khởi phát", ["Đột ngột", "Từ từ"])
            progression = st.selectbox("Diễn tiến", ["Nặng dần", "Ổn định", "Giảm"])

        with col3:
            st.subheader("🔍 Triệu chứng")
            chest_pain = st.checkbox("Đau ngực cấp")
            dyspnea = st.checkbox("Khó thở")
            trauma = st.checkbox("Chấn thương")
            pain_level = st.select_slider("Mức độ đau (VAS)", options=list(range(11)), value=0)

            st.markdown("**Context mở rộng (để chuyển khoa & protocol chuẩn hơn)**")
            fast_stroke = st.checkbox("FAST (+) nghi đột quỵ (méo miệng/yếu tay/nói khó)")
            bleeding = st.checkbox("Chảy máu (ngoài / nôn ra máu / phân đen)")
            abdominal_pain = st.checkbox("Đau bụng cấp")
            pregnancy = st.checkbox("Thai kỳ")
            infection_suspected = st.checkbox("Nghi nhiễm trùng (sốt/ớn lạnh/lơ mơ)")
            anaphylaxis = st.checkbox("Nghi phản vệ (phù mặt/khò khè/mề đay)")
            poisoning_overdose = st.checkbox("Nghi ngộ độc / quá liều")

        submit = st.form_submit_button("PHÂN LOẠI NGAY", type="primary", use_container_width=True)

    if submit:
        p = Patient(
            age=int(age), hr=int(hr), sbp=int(sbp), spo2=int(spo2), rr=int(rr), temp=float(temp),
            gcs_e=int(e), gcs_v=int(v), gcs_m=int(m),
            chest_pain=bool(chest_pain), dyspnea=bool(dyspnea), trauma=bool(trauma), pain_level=int(pain_level),
            onset=str(onset), progression=str(progression),
            fast_stroke=bool(fast_stroke), bleeding=bool(bleeding), abdominal_pain=bool(abdominal_pain),
            pregnancy=bool(pregnancy), infection_suspected=bool(infection_suspected),
            anaphylaxis=bool(anaphylaxis), poisoning_overdose=bool(poisoning_overdose)
        )

        ok, issues = validate_inputs(p)
        if issues:
            st.warning("Kiểm tra dữ liệu:\n- " + "\n- ".join(issues))
        if not ok:
            st.stop()

        g = gcs_total(p)
        si = calculate_shock_index(p.hr, p.sbp)
        ews = calculate_ews(p.hr, p.rr, p.sbp, p.temp, p.spo2)
        flags = red_flags(p, si, ews)

        risk, u, contrib_sorted, preds = ensemble_predict_with_explain(p)
        triage, color, note = triage_decision(flags, ews, risk, u, p)
        dept, dept_reason = recommend_department(p, triage, flags)
        actions = protocol_actions(dept, triage, p)

        st.markdown(f"<div class='triage-header' style='background-color:{color};'><h2>{triage}</h2></div>", unsafe_allow_html=True)
        st.caption(note)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("EWS", ews)
        c2.metric("Shock Index", si)
        c3.metric("GCS", f"{g}/15")
        c4.metric("Risk (AI)", f"{risk*100:.1f}%")
        c5.metric("Uncertainty (σ)", f"{u:.3f}")

        if flags:
            st.error("⚠️ **Red flags (Luật an toàn):** " + ", ".join(flags))

        st.markdown("### 🏥 Đề xuất chuyển khoa")
        st.write(f"**{dept}**")
        st.caption(f"Lý do: {dept_reason}")

        st.markdown("### 🧾 Protocol / Hành động gợi ý")
        for a in actions[:10]:
            st.write("• " + a)

        st.markdown("### 🔎 Lý do hỗ trợ quyết định (nhiều dữ kiện cho bác sĩ)")
        reasons = decision_support_reasons(p, triage, flags, ews, si, risk, u, dept)
        for r in reasons[:12]:
            st.write("• " + r)

        st.markdown("### 🔍 Lý do nổi bật (AI) – yếu tố tác động mạnh")
        st.write("• " + "\n• ".join(top_reasons(contrib_sorted)))

        sbar = (
            f"SBAR: BN {p.age}t. GCS {g}/15. HR {p.hr}. SBP {p.sbp}. RR {p.rr}. "
            f"SpO2 {p.spo2}%. Temp {p.temp}. EWS {ews}, SI {si}. "
            f"Risk {risk*100:.1f}%, Unc {u:.3f}. "
            f"Phân loại: {triage}. Chuyển khoa: {dept}."
        )
        st.text_area("Tóm tắt (SBAR):", sbar)

        # Save logs
        row = {
            "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Tuổi": p.age, "HR": p.hr, "SBP": p.sbp, "SpO2": p.spo2, "RR": p.rr, "Temp": p.temp,
            "GCS": g, "E": p.gcs_e, "V": p.gcs_v, "M": p.gcs_m,
            "Đau ngực": p.chest_pain, "Khó thở": p.dyspnea, "Chấn thương": p.trauma, "VAS": p.pain_level,
            "Khởi phát": p.onset, "Diễn tiến": p.progression,
            "FAST": p.fast_stroke, "Chảy máu": p.bleeding, "Đau bụng": p.abdominal_pain,
            "Thai kỳ": p.pregnancy, "Nghi nhiễm": p.infection_suspected,
            "Phản vệ": p.anaphylaxis, "Ngộ độc": p.poisoning_overdose,
            "EWS": ews, "ShockIndex": si,
            "Risk": risk, "Uncertainty": u, "UncLevel": uncertainty_level(u),
            "RedFlags": ", ".join(flags),
            "Phân loại": triage,
            "Khoa đề xuất": dept,
            "Lý do chuyển khoa": dept_reason,
            "Ghi chú": note,
            "SBAR": sbar,
            "AppVersion": APP_VERSION
        }
        st.session_state["logs"].append(row)
        st.session_state["last_case"] = {"row": row, "contrib": contrib_sorted, "preds": preds, "actions": actions, "reasons": reasons}

        st.markdown("<div class='small-note'>⚠️ Demo nghiên cứu/học thuật. Quyết định cuối cùng thuộc bác sĩ.</div>", unsafe_allow_html=True)

# -------------------------
# TAB 2: DASHBOARD
# -------------------------
with tab2:
    if st.session_state["logs"]:
        df = pd.DataFrame(st.session_state["logs"])
        colA, colB = st.columns([1, 1])
        with colA:
            st.subheader("Tỷ lệ theo phân loại")
            st.bar_chart(df["Phân loại"].value_counts())
        with colB:
            st.subheader("Tỷ lệ theo khoa đề xuất")
            st.bar_chart(df["Khoa đề xuất"].value_counts())

        st.markdown("---")
        st.subheader("Bảng tổng quan")
        show_cols = ["Thời gian", "Phân loại", "EWS", "ShockIndex", "GCS", "Risk", "Uncertainty", "Khoa đề xuất"]
        st.dataframe(df[show_cols], use_container_width=True, height=360)
    else:
        st.info("Chưa có dữ liệu. Vào tab **Tiếp nhận** để nhập ca.")

# -------------------------
# TAB 3: LOGS/EXPORT + DEEP EXPLAIN
# -------------------------
with tab3:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📑 Nhật ký (Audit trail)")
        if st.session_state["logs"]:
            df = pd.DataFrame(st.session_state["logs"])
            st.dataframe(df, use_container_width=True, height=420)
            st.download_button(
                "⬇️ Tải CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="triage_logs.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Chưa có log.")

    with right:
        st.subheader("🔍 Giải thích chuyên sâu (Case gần nhất)")
        case = st.session_state.get("last_case")
        if not case:
            st.info("Chưa có ca nào. Vào tab **Tiếp nhận** và bấm **PHÂN LOẠI NGAY**.")
        else:
            row = case["row"]
            st.markdown("### 1) Luật an toàn (Hard rules)")
            if row.get("RedFlags"):
                st.error("Kích hoạt: " + row["RedFlags"])
            else:
                st.success("Không kích hoạt red flags.")

            st.markdown("### 2) Điểm lâm sàng")
            st.write(f"- GCS: **{row['GCS']}/15**")
            st.write(f"- EWS: **{row['EWS']}**")
            st.write(f"- Shock Index: **{row['ShockIndex']}**")

            st.markdown("### 3) AI Risk + lý do")
            st.write(f"- Risk: **{row['Risk']*100:.1f}%**")
            st.write(f"- Uncertainty σ: **{row['Uncertainty']:.3f}** ({row['UncLevel']})")
            with st.expander("Bảng đóng góp đặc trưng (giải thích sâu)"):
                top = list(case["contrib"].items())[:14]
                table = [{"feature": k, "label": FEATURE_LABELS.get(k, k), "contribution": float(v)} for k, v in top]
                st.dataframe(pd.DataFrame(table), use_container_width=True)

            st.markdown("### 4) Lý do hỗ trợ quyết định (doctor-facing)")
            for r in case["reasons"][:14]:
                st.write("• " + r)

            st.markdown("### 5) Chuyển khoa + protocol")
            st.write(f"- Khoa: **{row['Khoa đề xuất']}**")
            st.caption("Lý do: " + row["Lý do chuyển khoa"])
            for a in case["actions"][:10]:
                st.write("• " + a)

            with st.expander("SBAR"):
                st.text(row["SBAR"])
