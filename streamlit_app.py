import base64
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# NEW libs
import qrcode
from PIL import Image
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# =========================
# CONFIG
# =========================
APP_VERSION = "v9.0 – High‑Trust + ESI + EWS Alert + Trend + Explainability + PDF+QR + CodeBlue"
MODEL_NOTE = "Safety-first: Red flags > Clinical protocol (EWS/ESI) > AI Risk+Uncertainty (HITL). Explainability = contribution-based (not clinical SHAP)."

st.set_page_config(page_title="Smart Triage AI Pro", layout="wide", page_icon="🚑")

st.markdown(
    """
<style>
.main { background-color: #0b1220; }
.block-container { padding-top: 1.2rem; }

/* Metric visibility on dark theme */
[data-testid="stMetric"] {
    background-color: #111827 !important;
    color: #F9FAFB !important;
    border-radius: 12px;
    padding: 16px;
    border: 1px solid #1F2937;
}
[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 0.9rem; }
[data-testid="stMetricValue"] { color: #F9FAFB !important; font-size: 1.6rem; font-weight: 700; }

.triage-header { text-align:center; padding:18px; border-radius:12px; color:white; margin:8px 0 18px 0; }
.small-note { color:#9CA3AF; font-size:0.9rem; }
.box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 4px solid #38bdf8;
    padding: 14px 16px;
    border-radius: 12px;
}

/* Big Code Blue button */
div.stButton > button.codeblue {
    background: #ef4444 !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    border: 1px solid #7f1d1d !important;
    padding: 14px 16px !important;
    width: 100% !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# DATA MODEL
# =========================
@dataclass
class Patient:
    age: int
    hr: int
    sbp: int
    spo2: int
    rr: int
    temp: float
    gcs_e: int
    gcs_v: int
    gcs_m: int
    chest_pain: bool
    dyspnea: bool
    trauma: bool
    pain_level: int
    onset: str
    progression: str
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
def gcs_total(p: Patient) -> int:
    return p.gcs_e + p.gcs_v + p.gcs_m


def calculate_shock_index(hr: int, sbp: int) -> float:
    return round(hr / sbp, 2) if sbp > 0 else 0.0


def calculate_ews(hr: int, rr: int, sbp: int, temp: float, spo2: int) -> int:
    score = 0
    if hr > 110 or hr < 50: score += 2
    if rr > 24 or rr < 10: score += 2
    if sbp < 90 or sbp > 180: score += 2
    if temp > 38.5 or temp < 35.5: score += 1
    if spo2 < 94: score += 3
    return score


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
        soft.append("Thai kỳ + tuổi rất nhỏ (kiểm tra lại).")
    return (len(hard) == 0), hard + soft


# =========================
# HARD SAFETY (RED FLAGS)
# =========================
def red_flags(p: Patient, si: float, ews: int):
    g = gcs_total(p)
    flags = []
    if p.anaphylaxis: flags.append("Nghi sốc phản vệ")
    if g <= 8: flags.append("Hôn mê nặng (GCS ≤ 8)")
    if p.fast_stroke: flags.append("FAST dương tính (nghi đột quỵ)")
    if p.spo2 < 90: flags.append("Suy hô hấp nặng (SpO₂ < 90%)")
    if p.sbp < 90: flags.append("Sốc / tụt huyết áp (SBP < 90)")
    if si > 1.0: flags.append(f"Shock Index nguy hiểm ({si})")
    if p.rr >= 30: flags.append("Thở nhanh nặng (RR ≥ 30)")
    if p.hr >= 140: flags.append("Mạch nhanh nặng (HR ≥ 140)")
    if p.bleeding and (p.sbp < 100 or p.hr > 110): flags.append("Chảy máu + huyết động xấu")
    if p.poisoning_overdose and g <= 12: flags.append("Nghi ngộ độc + giảm tri giác")
    if ews >= 7: flags.append("EWS rất cao (≥ 7)")
    return flags


# =========================
# ESI (ESI-lite)
# =========================
def estimate_resources(p: Patient) -> int:
    r = 0
    if p.chest_pain: r += 2
    if p.dyspnea or p.spo2 < 94: r += 2
    if p.trauma: r += 2
    if p.bleeding: r += 2
    if p.abdominal_pain: r += 1
    if p.infection_suspected: r += 1
    if p.poisoning_overdose: r += 2
    if p.pregnancy: r += 1
    return r


def esi_level(p: Patient, flags: list, ews: int):
    if flags:
        return 1, "ESI‑1: cần can thiệp cứu sống ngay (red flags)."
    if p.fast_stroke or p.anaphylaxis or p.chest_pain or p.dyspnea or ews >= 3:
        return 2, "ESI‑2: nguy cơ cao/không được chậm (triệu chứng/điểm cảnh báo)."
    res = estimate_resources(p)
    if res >= 2: return 3, f"ESI‑3: ổn định nhưng cần ≥2 resources (ước lượng: {res})."
    if res == 1: return 4, "ESI‑4: ổn định, cần 1 resource."
    return 5, "ESI‑5: ổn định, hầu như không cần resource."


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
    "pregnancy": "Thai kỳ",
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
# EXPLAINABILITY (Feature Importance %)
# =========================
def feature_importance_percent(contrib_sorted: dict, top_n: int = 8) -> pd.DataFrame:
    """
    Convert contribution magnitudes into % so BGK can see:
    'RR contributes 40%...'
    Note: this is a transparent contribution-based explanation (not SHAP).
    """
    items = list(contrib_sorted.items())
    mags = np.array([abs(v) for _, v in items], dtype=float)
    total = float(mags.sum()) if mags.sum() > 0 else 1.0
    rows = []
    for k, v in items[:top_n]:
        pct = abs(v) / total * 100.0
        rows.append({"Feature": k, "Yếu tố": FEATURE_LABELS.get(k, k), "Đóng góp (%)": pct, "Hướng": "Tăng nguy cơ" if v > 0 else "Giảm nguy cơ"})
    return pd.DataFrame(rows)


# =========================
# ALERT + CODE BLUE
# =========================
def should_alert(flags: list, ews: int) -> bool:
    return bool(flags) or (ews >= 5)


def is_code_blue(p: Patient) -> bool:
    """
    Ngưỡng cực nguy kịch (demo):
    - SBP < 80
    - SpO2 < 85
    - GCS <= 6
    - RR < 6 hoặc RR > 35
    """
    g = gcs_total(p)
    return (p.sbp < 80) or (p.spo2 < 85) or (g <= 6) or (p.rr < 6) or (p.rr > 35)


def send_alert(message: str) -> bool:
    # DEMO notifier: simulate
    return True


# =========================
# TRIAGE DECISION
# =========================
def triage_decision(flags: list, ews: int, risk: float, u: float, p: Patient):
    if flags:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", "Luật an toàn kích hoạt: " + ", ".join(flags)

    if ews >= 5:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", f"EWS cao (≥5): {ews}. Ưu tiên đánh giá ngay."

    if (
        ews >= 3
        or p.chest_pain
        or p.pain_level >= 7
        or p.fast_stroke
        or p.anaphylaxis
        or p.bleeding
        or p.poisoning_overdose
    ):
        note = f"Ưu tiên theo triệu chứng/điểm: EWS={ews}."
        if uncertainty_level(u) == "CAO":
            note += " Uncertainty CAO → cần bác sĩ xác nhận/đo lại."
        return "🟡 VÀNG (ƯU TIÊN)", "#FFA500", note

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
# DEPARTMENT + PROTOCOL (rút gọn)
# =========================
def recommend_department(p: Patient, triage: str, flags: list):
    is_peds = p.age < 16
    g = gcs_total(p)

    if flags or ("🔴" in triage):
        if is_peds: return "Cấp cứu/Hồi sức → Nhi", "Nguy kịch + tuổi nhi."
        if p.anaphylaxis: return "Cấp cứu/Hồi sức", "Phản vệ: ưu tiên ABC."
        if p.fast_stroke or g <= 12: return "Cấp cứu/Hồi sức → Thần kinh", "Giảm tri giác/FAST (+)."
        if p.bleeding: return "Cấp cứu/Hồi sức → Ngoại/Tiêu hoá", "Chảy máu: hồi sức."
        if p.chest_pain: return "Cấp cứu/Hồi sức → Tim mạch", "Đau ngực nguy kịch."
        if p.dyspnea or p.spo2 < 94: return "Cấp cứu/Hồi sức → Hô hấp", "Khó thở/SpO₂ giảm."
        return "Cấp cứu/Hồi sức", "Ổn định ABC trước."

    if is_peds: return "Nhi", "Tuổi < 16."
    if p.pregnancy: return "Sản", "Thai kỳ."
    if p.fast_stroke or g <= 13: return "Thần kinh", "Nghi đột quỵ/tri giác giảm."
    if p.trauma: return "Ngoại/Chấn thương", "Chấn thương."
    if p.chest_pain: return "Tim mạch", "Đau ngực."
    if p.dyspnea or p.spo2 < 94: return "Hô hấp", "Khó thở/SpO₂ giảm."
    if p.infection_suspected: return "Nội/Nhiễm", "Nghi nhiễm trùng."
    return "Cấp cứu/Nội tổng quát", "Không có cụm nổi bật."


def protocol_actions(dept: str, triage: str, p: Patient):
    actions = []
    if "🔴" in triage:
        actions += ["ABC + monitor + đường truyền", "Bác sĩ đánh giá ngay", "Đo lại sinh hiệu liên tục"]
    elif "🟡" in triage:
        actions += ["Khám ưu tiên", "Theo dõi sát", "Cận lâm sàng theo triệu chứng"]
    else:
        actions += ["Theo dõi cơ bản", "Tư vấn và dặn tái khám"]

    if "Tim mạch" in dept: actions += ["ECG sớm", "Men tim theo protocol"]
    if "Thần kinh" in dept: actions += ["Đường huyết", "CT theo quy trình đột quỵ"]
    if "Hô hấp" in dept: actions += ["Oxy", "X-quang phổi/khí máu nếu cần"]
    if "Ngoại" in dept: actions += ["ABCDE", "Kiểm soát chảy máu/bất động"]
    if "Sản" in dept: actions += ["Đánh giá mẹ và thai"]
    if p.anaphylaxis: actions += ["Phác đồ phản vệ"]
    if p.fast_stroke: actions += ["Kích hoạt stroke pathway"]
    # dedup
    out, seen = [], set()
    for a in actions:
        if a not in seen:
            seen.add(a); out.append(a)
    return out


# =========================
# TREND
# =========================
def detect_worsening_trend(df: pd.DataFrame):
    if len(df) < 3:
        return None
    last3 = df.tail(3)
    reasons = []
    if last3["EWS"].iloc[-1] > last3["EWS"].iloc[0]: reasons.append("EWS tăng")
    if last3["SpO2"].iloc[-1] < last3["SpO2"].iloc[0]: reasons.append("SpO₂ giảm")
    if last3["SBP"].iloc[-1] < last3["SBP"].iloc[0]: reasons.append("SBP giảm")
    if last3["GCS"].iloc[-1] < last3["GCS"].iloc[0]: reasons.append("GCS giảm")
    return ("Xu hướng xấu: " + ", ".join(reasons)) if reasons else None


# =========================
# PDF EXPORT
# =========================
def make_pdf_bytes(title: str, lines: list[str]) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 60

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 26

    c.setFont("Helvetica", 10)
    for line in lines:
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 60
        c.drawString(50, y, line[:120])
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# QR SYNC
# =========================
def make_case_payload(case_row: dict) -> str:
    """
    Encode a minimal JSON payload for QR/transfer.
    Keep it small & stable (no huge arrays).
    """
    minimal = {k: case_row.get(k) for k in [
        "Thời gian","Tuổi","HR","SBP","SpO2","RR","Temp","GCS","E","V","M",
        "Đau ngực","Khó thở","Chấn thương","VAS","Khởi phát","Diễn tiến",
        "FAST","Chảy máu","Đau bụng","Thai kỳ","Nghi nhiễm","Phản vệ","Ngộ độc",
        "EWS","ShockIndex","ESI","Risk","Uncertainty","UncLevel","RedFlags",
        "ALERT","Phân loại","Khoa đề xuất","Lý do chuyển khoa","Ghi chú","SBAR","AppVersion"
    ]}
    js = json.dumps(minimal, ensure_ascii=False)
    b64 = base64.urlsafe_b64encode(js.encode("utf-8")).decode("utf-8")
    return b64


def payload_to_case(b64: str) -> dict:
    js = base64.urlsafe_b64decode(b64.encode("utf-8")).decode("utf-8")
    return json.loads(js)


def make_qr_image(data: str) -> Image.Image:
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img


# =========================
# STATE
# =========================
def init_state():
    if "logs" not in st.session_state: st.session_state["logs"] = []
    if "last_case" not in st.session_state: st.session_state["last_case"] = None
    if "vitals_series" not in st.session_state: st.session_state["vitals_series"] = []
    if "enable_notify" not in st.session_state: st.session_state["enable_notify"] = False
    if "code_blue_events" not in st.session_state: st.session_state["code_blue_events"] = []


init_state()

# =========================
# SIDEBAR (Offline + Notify)
# =========================
st.sidebar.subheader("⚙️ Cấu hình")
st.session_state["enable_notify"] = st.sidebar.checkbox("Bật gửi cảnh báo (demo)", value=st.session_state["enable_notify"])
st.sidebar.caption("Muốn gửi thật (Telegram/Email/Webhook) mình cắm token cho bạn.")

st.sidebar.markdown("---")
st.sidebar.subheader("📴 Chế độ Offline (thực tế)")
st.sidebar.write(
    "• Streamlit **chạy offline tốt** khi bạn chạy local/intranet.\n"
    "• Nếu deploy Cloud thì cần internet.\n"
    "• Bài thi: bạn trình bày mô hình triển khai **Laptop cấp cứu / Server nội bộ bệnh viện**."
)

# =========================
# APP HEADER
# =========================
st.title("🏥 Smart Triage AI Pro – Hospital‑Wide")
st.caption(f"{APP_VERSION} | {MODEL_NOTE}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Tiếp nhận",
    "📊 Dashboard (Trend)",
    "🧠 Explainability + PDF/QR",
    "🔄 Nhập ca từ QR/Payload"
])


# =========================
# TAB 1: INTAKE
# =========================
with tab1:
    # CODE BLUE manual button (always visible)
    st.markdown("### 🚨 CODE BLUE")
    col_cb1, col_cb2 = st.columns([2, 3])
    with col_cb1:
        code_blue_manual = st.button("KÍCH HOẠT CODE BLUE (TOÀN VIỆN)", type="primary")
        # style class hack
        st.markdown("""
        <script>
        const btns = window.parent.document.querySelectorAll('button[kind="primary"]');
        btns.forEach(b => { if (b.innerText.includes("CODE BLUE")) b.classList.add("codeblue"); });
        </script>
        """, unsafe_allow_html=True)
    with col_cb2:
        st.caption("Dùng khi sinh hiệu tụt cực nặng / ngưng tuần hoàn nghi ngờ. (Demo: chỉ log + cảnh báo UI)")

    if code_blue_manual:
        st.session_state["code_blue_events"].append({"time": datetime.now().isoformat(), "type": "MANUAL"})
        st.error("🚨 CODE BLUE đã kích hoạt (manual) — demo log đã ghi lại.")

    st.markdown("---")

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
            st.subheader("🔍 Triệu chứng + Context")
            chest_pain = st.checkbox("Đau ngực cấp")
            dyspnea = st.checkbox("Khó thở")
            trauma = st.checkbox("Chấn thương")
            pain_level = st.select_slider("Mức độ đau (VAS)", options=list(range(11)), value=0)

            fast_stroke = st.checkbox("FAST (+) nghi đột quỵ")
            bleeding = st.checkbox("Chảy máu")
            abdominal_pain = st.checkbox("Đau bụng cấp")
            pregnancy = st.checkbox("Thai kỳ")
            infection_suspected = st.checkbox("Nghi nhiễm trùng")
            anaphylaxis = st.checkbox("Nghi phản vệ")
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

        esi, esi_note = esi_level(p, flags, ews)
        risk, u, contrib_sorted, preds = ensemble_predict_with_explain(p)
        triage, color, note = triage_decision(flags, ews, risk, u, p)
        dept, dept_reason = recommend_department(p, triage, flags)
        actions = protocol_actions(dept, triage, p)

        alert = should_alert(flags, ews)
        blue = is_code_blue(p)

        st.markdown(f"<div class='triage-header' style='background-color:{color};'><h2>{triage}</h2></div>", unsafe_allow_html=True)
        st.caption(note)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("EWS", ews)
        c2.metric("Shock Index", si)
        c3.metric("GCS", f"{g}/15")
        c4.metric("ESI (tham khảo)", f"ESI-{esi}")
        c5.metric("Risk (AI)", f"{risk*100:.1f}%")
        c6.metric("Uncertainty (σ)", f"{u:.3f}")
        st.caption(esi_note)

        if flags:
            st.error("⚠️ Red flags: " + ", ".join(flags))

        if blue:
            st.error("🛑 NGƯỠNG CODE BLUE (AUTO): Sinh hiệu cực nguy kịch! (Demo: bật cảnh báo + ghi log)")
            st.session_state["code_blue_events"].append({"time": datetime.now().isoformat(), "type": "AUTO", "SBP": p.sbp, "SpO2": p.spo2, "GCS": g})
            if st.session_state["enable_notify"]:
                send_alert(f"[CODE BLUE] SBP={p.sbp} SpO2={p.spo2} GCS={g} | Dept={dept}")
                st.success("✅ Đã gửi CODE BLUE (demo).")

        if alert:
            st.error("🚨 CẢNH BÁO SỚM (EWS/Red‑flags): ưu tiên xử trí ngay!")
            if st.session_state["enable_notify"]:
                send_alert(f"[ALERT] {triage} | EWS={ews} | SBP={p.sbp} | SpO2={p.spo2} | GCS={g} | Dept={dept}")
                st.success("✅ Đã gửi cảnh báo (demo).")

        st.markdown("### 🏥 Đề xuất chuyển khoa")
        st.write(f"**{dept}**")
        st.caption(f"Lý do: {dept_reason}")

        st.markdown("### 🧾 Protocol / Hành động gợi ý")
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        for a in actions[:12]:
            st.write("• " + a)
        st.markdown("</div>", unsafe_allow_html=True)

        sbar = (
            f"SBAR: BN {p.age}t. GCS {g}/15. HR {p.hr}. SBP {p.sbp}. RR {p.rr}. "
            f"SpO2 {p.spo2}%. Temp {p.temp}. EWS {ews}, SI {si}. ESI-{esi}. "
            f"Risk {risk*100:.1f}%, Unc {u:.3f}. "
            f"Phân loại: {triage}. Chuyển khoa: {dept}."
        )
        st.text_area("Tóm tắt (SBAR):", sbar)

        # trend store
        st.session_state["vitals_series"].append({
            "time": datetime.now(),
            "HR": p.hr, "SBP": p.sbp, "SpO2": p.spo2, "RR": p.rr, "Temp": p.temp,
            "GCS": g, "EWS": ews, "ESI": esi
        })

        # logs
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
            "ESI": esi,
            "Risk": risk, "Uncertainty": u, "UncLevel": uncertainty_level(u),
            "RedFlags": ", ".join(flags),
            "ALERT": alert,
            "CODE_BLUE_AUTO": blue,
            "Phân loại": triage,
            "Khoa đề xuất": dept,
            "Lý do chuyển khoa": dept_reason,
            "Ghi chú": note,
            "SBAR": sbar,
            "AppVersion": APP_VERSION
        }

        st.session_state["logs"].append(row)
        st.session_state["last_case"] = {"row": row, "contrib_sorted": contrib_sorted}

        st.markdown("<div class='small-note'>⚠️ Demo học thuật. Quyết định cuối cùng thuộc bác sĩ.</div>", unsafe_allow_html=True)


# =========================
# TAB 2: DASHBOARD (TREND)
# =========================
with tab2:
    st.subheader("📈 Xu hướng sinh hiệu theo thời gian (Trend)")
    if st.session_state["vitals_series"]:
        tdf = pd.DataFrame(st.session_state["vitals_series"]).sort_values("time").reset_index(drop=True)

        st.line_chart(tdf.set_index("time")[["HR", "SBP", "SpO2", "RR", "Temp"]])
        st.line_chart(tdf.set_index("time")[["GCS", "EWS"]])

        msg = detect_worsening_trend(tdf)
        if msg:
            st.warning("⚠️ " + msg)
        else:
            st.success("✅ Chưa phát hiện xu hướng xấu rõ rệt (trên 3 lần đo gần nhất).")
    else:
        st.info("Chưa có dữ liệu trend. Nhập ca vài lần để có đồ thị.")


# =========================
# TAB 3: EXPLAINABILITY + PDF/QR
# =========================
with tab3:
    st.subheader("🧠 AI Explainability (Feature Importance) + PDF/QR Đồng bộ")

    case = st.session_state.get("last_case")
    if not case:
        st.info("Chưa có ca. Vào tab Tiếp nhận → PHÂN LOẠI NGAY.")
    else:
        row = case["row"]
        contrib_sorted = case["contrib_sorted"]

        st.markdown("### 1) Feature Importance (dạng % để trả lời BGK “vì sao ra 1.8%?”)")
        df_imp = feature_importance_percent(contrib_sorted, top_n=10)
        st.dataframe(df_imp, use_container_width=True, height=360)
        st.bar_chart(df_imp.set_index("Yếu tố")[["Đóng góp (%)"]])

        st.caption(
            "Giải thích: % đóng góp được chuẩn hoá từ |contribution| của các đặc trưng trong mô hình demo (giải thích được, audit được). "
            "Không phải SHAP lâm sàng, nhưng đủ minh bạch để trả lời BGK."
        )

        st.markdown("---")
        st.markdown("### 2) Xuất PDF (bệnh án tóm tắt + SBAR)")
        lines = [
            f"Thời gian: {row['Thời gian']}",
            f"Phân loại: {row['Phân loại']} | EWS={row['EWS']} | SI={row['ShockIndex']} | GCS={row['GCS']} | ESI={row['ESI']}",
            f"Khoa đề xuất: {row['Khoa đề xuất']} (Lý do: {row['Lý do chuyển khoa']})",
            f"AI Risk: {row['Risk']*100:.1f}% | Uncertainty: {row['Uncertainty']:.3f} ({row['UncLevel']})",
            f"RedFlags: {row['RedFlags']}",
            f"ALERT: {row['ALERT']} | CODE_BLUE_AUTO: {row['CODE_BLUE_AUTO']}",
            "---- Vitals ----",
            f"HR={row['HR']} | SBP={row['SBP']} | SpO2={row['SpO2']} | RR={row['RR']} | Temp={row['Temp']}",
            "---- SBAR ----",
            row["SBAR"],
            "---- Feature Importance (Top) ----",
        ]
        for _, r in df_imp.head(8).iterrows():
            lines.append(f"{r['Yếu tố']}: {r['Đóng góp (%)']:.1f}% ({r['Hướng']})")

        pdf_bytes = make_pdf_bytes("SMART TRIAGE AI – PDF SUMMARY", lines)
        st.download_button(
            "⬇️ Tải PDF",
            data=pdf_bytes,
            file_name="triage_case_summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("### 3) Tạo QR/Payload để đồng bộ ca bệnh")
        payload = make_case_payload(row)

        st.caption("Cách dùng: máy khác mở app → tab “Nhập ca từ QR/Payload” → dán payload (hoặc quét QR nếu bạn tích hợp camera sau).")
        st.code(payload[:220] + ("..." if len(payload) > 220 else ""))

        qr_img = make_qr_image(payload)
        st.image(qr_img, caption="QR đồng bộ ca bệnh (payload base64)", width=260)

        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.download_button("⬇️ Tải QR (PNG)", data=buf.getvalue(), file_name="case_qr.png", mime="image/png", use_container_width=True)


# =========================
# TAB 4: IMPORT CASE FROM PAYLOAD
# =========================
with tab4:
    st.subheader("🔄 Nhập ca từ QR/Payload (Interoperability)")
    st.caption("Dán payload base64 (từ QR/tab Explainability) để load lại dữ liệu trên máy khác.")

    payload_in = st.text_area("Payload (base64)", height=160)
    if st.button("📥 LOAD CASE", use_container_width=True):
        try:
            obj = payload_to_case(payload_in.strip())
            # Add to logs
            st.session_state["logs"].append(obj)
            st.success("✅ Đã import ca vào logs.")
            st.json(obj)
        except Exception as e:
            st.error(f"Payload không hợp lệ: {e}")

    st.markdown("---")
    st.subheader("📑 Logs/Export CSV + CodeBlue events")
    if st.session_state["logs"]:
        df = pd.DataFrame(st.session_state["logs"])
        st.dataframe(df, use_container_width=True, height=380)
        st.download_button(
            "⬇️ Tải CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="triage_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state["code_blue_events"]:
        st.markdown("### 🚨 Code Blue events (audit)")
        st.dataframe(pd.DataFrame(st.session_state["code_blue_events"]), use_container_width=True)
