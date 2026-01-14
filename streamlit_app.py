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
.main { background-color: #f8f9fa; }
.block-container { padding-top: 1.2rem; }
.stMetric { background-color: #ffffff; padding: 14px; border-radius: 10px;
           box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.triage-header { text-align: center; padding: 18px; border-radius: 12px; color: white; margin: 8px 0 18px 0; }
.small-note { color: #6c757d; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

APP_VERSION = "v5.0 – Combined (EWS+GCS+SI + Risk/Uncertainty + Dept + Explain + Audit)"
MODEL_NOTE = "Safety-first: Red flags > EWS > AI Risk+Uncertainty (Human-in-the-loop)."

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

    # Neuro
    gcs_e: int
    gcs_v: int
    gcs_m: int

    # Symptoms/context
    chest_pain: bool
    dyspnea: bool
    trauma: bool
    pain_level: int  # VAS 0-10

    onset: str        # "Đột ngột" / "Từ từ"
    progression: str  # "Nặng dần" / "Ổn định" / "Giảm"

# =========================
# CLINICAL UTILITIES
# =========================
def calculate_shock_index(hr, sbp):
    return round(hr / sbp, 2) if sbp > 0 else 0.0

def calculate_ews(hr, rr, sbp, temp, spo2):
    """Early Warning Score (đơn giản hoá cho demo tổng quát)"""
    score = 0
    if hr > 110 or hr < 50: score += 2
    if rr > 24 or rr < 10: score += 2
    if sbp < 90 or sbp > 180: score += 2
    if temp > 38.5 or temp < 35.5: score += 1
    if spo2 < 94: score += 3
    return score

# =========================
# SAFETY / VALIDATION
# =========================
def validate_inputs(p: Patient):
    hard, soft = [], []
    if not (0 <= p.age <= 120): hard.append("Tuổi ngoài phạm vi 0–120.")
    if not (20 <= p.hr <= 250): hard.append("HR ngoài phạm vi 20–250.")
    if not (40 <= p.sbp <= 250): hard.append("SBP ngoài phạm vi 40–250.")
    if not (50 <= p.spo2 <= 100): hard.append("SpO₂ ngoài phạm vi 50–100%.")
    if not (5 <= p.rr <= 60): hard.append("RR ngoài phạm vi 5–60.")
    if not (34.0 <= p.temp <= 42.0): hard.append("Nhiệt độ ngoài phạm vi 34–42°C.")
    gcs = p.gcs_e + p.gcs_v + p.gcs_m
    if not (3 <= gcs <= 15): hard.append("GCS không hợp lệ.")

    # soft consistency hints
    if p.spo2 < 88 and not p.dyspnea:
        soft.append("SpO₂ rất thấp nhưng chưa tick 'Khó thở' (kiểm tra lại).")

    return (len(hard) == 0), hard + soft

def red_flags(p: Patient, si: float, ews: int):
    """Hard safety – ưu tiên tuyệt đối"""
    gcs = p.gcs_e + p.gcs_v + p.gcs_m
    flags = []

    if gcs <= 8: flags.append("Hôn mê nặng (GCS ≤ 8)")
    if p.spo2 < 90: flags.append("Suy hô hấp nặng (SpO₂ < 90%)")
    if p.sbp < 90: flags.append("Sốc / tụt huyết áp (SBP < 90)")
    if si > 1.0: flags.append(f"Shock Index nguy hiểm ({si})")
    if p.rr >= 30: flags.append("Thở nhanh nặng (RR ≥ 30)")
    if p.hr >= 140: flags.append("Mạch nhanh nặng (HR ≥ 140)")
    # EWS rất cao cũng coi là nguy kịch theo quy trình
    if ews >= 7: flags.append("EWS rất cao (≥ 7)")

    return flags

# =========================
# AI RISK + UNCERTAINTY (ENSEMBLE, NO TRAIN)
# =========================
def sigmoid(x: float) -> float:
    x = max(min(x, 40), -40)
    return 1.0 / (1.0 + math.exp(-x))

def features(p: Patient):
    gcs = p.gcs_e + p.gcs_v + p.gcs_m
    return {
        "age": float(p.age),
        "hr_excess": float(max(0, p.hr - 90)),
        "sbp_drop": float(max(0, 100 - p.sbp)),
        "spo2_drop": float(max(0, 95 - p.spo2)),
        "rr_excess": float(max(0, p.rr - 18)),
        "temp_excess": float(max(0, p.temp - 37.5)),
        "gcs_drop": float(max(0, 15 - gcs)),  # GCS giảm -> rủi ro tăng
        "chest_pain": float(int(p.chest_pain)),
        "dyspnea": float(int(p.dyspnea)),
        "trauma": float(int(p.trauma)),
        "pain_hi": float(int(p.pain_level >= 7)),
        "onset_sudden": float(int(p.onset == "Đột ngột")),
        "worsening": float(int(p.progression == "Nặng dần")),
    }

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
    "age": "Tuổi",
}

def ensemble_predict_with_explain(p: Patient):
    """
    Ensemble logistic: trả (mean_risk, std_uncertainty, contrib_sorted, preds)
    """
    base = {
        "b0": -7.0,
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
    }

    x = features(p)
    rng = np.random.default_rng(42)
    preds = []
    for _ in range(21):
        z = base["b0"] + rng.normal(0, 0.30)
        for k, v in x.items():
            z += base[k] * (1 + rng.normal(0, 0.10)) * v
        preds.append(sigmoid(z))

    arr = np.array(preds, dtype=float)
    mean_r = float(arr.mean())
    std_u = float(arr.std(ddof=1))

    # explain (không phải SHAP nhưng “giải thích được”)
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
# DECISION POLICY (Safety-first + HITL)
# =========================
def triage_decision(flags: list, ews: int, risk: float, u: float, p: Patient):
    """
    Priority order:
    1) Red flags -> RED
    2) EWS high -> RED/YELLOW
    3) Otherwise risk + uncertainty (HITL)
    """
    if flags:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", "Luật an toàn kích hoạt: " + ", ".join(flags)

    # EWS logic (protocol-friendly)
    if ews >= 5:
        return "🔴 ĐỎ (CẤP CỨU)", "#FF4B4B", f"EWS cao (≥5): {ews}. Ưu tiên đánh giá ngay."
    if ews >= 3 or p.chest_pain or p.pain_level >= 7:
        # vẫn check uncertainty: nếu uncertainty cao -> yêu cầu confirm
        note = f"EWS trung bình/triệu chứng ưu tiên: EWS={ews}."
        if uncertainty_level(u) == "CAO":
            note += " Uncertainty CAO → cần bác sĩ xác nhận/đo lại."
        return "🟡 VÀNG (ƯU TIÊN)", "#FFA500", note

    # AI risk + uncertainty
    base = triage_from_risk(risk)
    if base.startswith("🔴"):
        if uncertainty_level(u) == "CAO":
            return "🟡 VÀNG (REVIEW)", "#FFA500", "Risk cao nhưng Uncertainty CAO → không áp đặt ĐỎ, cần bác sĩ review."
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
    """
    Đề xuất khoa (tổng quát). Nếu ĐỎ/flags: Resus/ICU trước rồi định hướng.
    """
    is_peds = p.age < 16
    gcs = p.gcs_e + p.gcs_v + p.gcs_m

    if flags or ("🔴" in triage):
        if is_peds:
            return "Cấp cứu/Hồi sức (ưu tiên) → Nhi", "Nguy kịch + tuổi nhi."
        if p.trauma:
            return "Cấp cứu/Hồi sức (ưu tiên) → Ngoại/Chấn thương", "Nguy kịch + chấn thương."
        if p.chest_pain:
            return "Cấp cứu/Hồi sức (ưu tiên) → Tim mạch", "Nguy kịch + đau ngực."
        if p.dyspnea or p.spo2 < 94:
            return "Cấp cứu/Hồi sức (ưu tiên) → Hô hấp", "Nguy kịch + khó thở/SpO₂ giảm."
        if gcs <= 12:
            return "Cấp cứu/Hồi sức (ưu tiên) → Thần kinh", "Nguy kịch + tri giác giảm."
        return "Cấp cứu/Hồi sức (ưu tiên)", "Ưu tiên ổn định ABC trước, sau đó phân khoa."

    # Non-red routing
    if is_peds:
        return "Nhi (hoặc Cấp cứu Nhi)", "Tuổi < 16."
    if p.trauma:
        return "Ngoại/Chấn thương chỉnh hình", "Chấn thương là triệu chứng chính."
    if p.chest_pain:
        return "Tim mạch", "Đau ngực → ưu tiên ECG/men tim theo quy trình."
    if p.dyspnea or p.spo2 < 94:
        return "Hô hấp", "Khó thở/SpO₂ giảm."
    if gcs <= 13 or (p.onset == "Đột ngột" and p.progression == "Nặng dần"):
        return "Thần kinh", "Tri giác giảm hoặc diễn tiến đáng ngại."
    if p.temp >= 38.5 and (p.hr >= 110 or p.rr >= 22):
        return "Nội tổng quát / Nhiễm (tuỳ BV)", "Gợi ý nhiễm trùng: sốt + đáp ứng viêm."
    return "Cấp cứu tổng quát / Nội tổng quát", "Không có cụm triệu chứng nổi trội."

# =========================
# EXPLANATION HELPERS
# =========================
def top_reasons(contrib_sorted, k=6):
    out = []
    for feat, val in list(contrib_sorted.items())[:k]:
        if abs(val) < 0.05:
            continue
        out.append(FEATURE_LABELS.get(feat, feat))
    return out if out else ["Không có yếu tố nổi bật"]

def uncertainty_reasons(p: Patient, ews: int, risk: float, u: float):
    reasons = []
    # Near thresholds / gray zone
    if 0.25 <= risk <= 0.45:
        reasons.append("Risk nằm vùng xám (gần ngưỡng Vàng).")
    if 0.60 <= risk <= 0.80:
        reasons.append("Risk gần ngưỡng Đỏ.")
    if 3 <= ews <= 5:
        reasons.append("EWS gần ngưỡng cảnh báo.")
    # Potential inconsistency
    if p.spo2 < 88 and not p.dyspnea:
        reasons.append("SpO₂ rất thấp nhưng không ghi nhận khó thở (mâu thuẫn).")
    if p.sbp < 90 and p.hr < 60:
        reasons.append("SBP thấp nhưng HR không tăng (cần kiểm tra đo lại).")
    # Missing context signals
    if p.pain_level == 0 and (p.chest_pain or p.trauma):
        reasons.append("VAS=0 nhưng có triệu chứng (cần xác nhận mức đau).")
    return reasons if reasons else ["Uncertainty thấp: các mô hình đồng thuận cao."]

def action_suggestions(triage: str, dept: str):
    if "🔴" in triage:
        return [
            "Ưu tiên ABC: đường thở – hô hấp – tuần hoàn.",
            "Theo dõi monitor, đo lại sinh hiệu sớm.",
            f"Chuyển/điều phối: {dept}.",
            "Bác sĩ đánh giá ngay."
        ]
    if "🟡" in triage:
        return [
            "Khám ưu tiên, theo dõi sát.",
            "Đo lại sinh hiệu nếu thay đổi triệu chứng.",
            f"Định hướng chuyên khoa: {dept}.",
            "Nếu nặng lên → nâng mức xử trí."
        ]
    return [
        "Theo dõi, tư vấn, đánh giá thêm nếu cần.",
        f"Định hướng: {dept}.",
        "Dặn tái khám nếu xuất hiện dấu hiệu nguy hiểm."
    ]

# =========================
# MAIN APP
# =========================
st.title("🏥 Smart Triage AI Pro – Hệ thống phân loại cấp cứu tổng quát")
st.caption(f"{APP_VERSION} | {MODEL_NOTE}")

if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "last_case" not in st.session_state:
    st.session_state["last_case"] = None

tab1, tab2, tab3 = st.tabs(["📝 Tiếp nhận", "📊 Dashboard", "📑 Nhật ký / Export"])

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
            gcs_total = e + v + m
            st.info(f"Tổng điểm GCS: {gcs_total}/15")

            onset = st.selectbox("Khởi phát", ["Đột ngột", "Từ từ"])
            progression = st.selectbox("Diễn tiến", ["Nặng dần", "Ổn định", "Giảm"])

        with col3:
            st.subheader("🔍 Triệu chứng")
            chest_pain = st.checkbox("Đau ngực cấp")
            dyspnea = st.checkbox("Khó thở")
            trauma = st.checkbox("Chấn thương")
            pain_level = st.select_slider("Mức độ đau (VAS)", options=list(range(11)), value=0)

        submit = st.form_submit_button("PHÂN LOẠI NGAY", type="primary", use_container_width=True)

    if submit:
        p = Patient(
            age=int(age), hr=int(hr), sbp=int(sbp), spo2=int(spo2), rr=int(rr), temp=float(temp),
            gcs_e=int(e), gcs_v=int(v), gcs_m=int(m),
            chest_pain=bool(chest_pain), dyspnea=bool(dyspnea), trauma=bool(trauma), pain_level=int(pain_level),
            onset=str(onset), progression=str(progression)
        )

        ok, issues = validate_inputs(p)
        if issues:
            st.warning("Kiểm tra dữ liệu:\n- " + "\n- ".join(issues))

        # core scores
        si = calculate_shock_index(p.hr, p.sbp)
        ews = calculate_ews(p.hr, p.rr, p.sbp, p.temp, p.spo2)
        flags = red_flags(p, si, ews)

        # AI
        risk, u, contrib_sorted, preds = ensemble_predict_with_explain(p)

        # Decision
        triage, color, note = triage_decision(flags, ews, risk, u, p)

        # Department
        dept, dept_reason = recommend_department(p, triage, flags)

        # Display header
        st.markdown(
            f"<div class='triage-header' style='background-color:{color};'><h2>{triage}</h2></div>",
            unsafe_allow_html=True
        )
        st.caption(note)

        # metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("EWS", ews)
        c2.metric("Shock Index", si)
        c3.metric("GCS", f"{gcs_total}/15")
        c4.metric("Risk (AI)", f"{risk*100:.1f}%")
        c5.metric("Uncertainty (σ)", f"{u:.3f}")

        if flags:
            st.error("⚠️ **Red flags (Luật an toàn):** " + ", ".join(flags))

        # Department recommendation
        st.markdown("### 🏥 Đề xuất chuyển khoa")
        st.write(f"**{dept}**")
        st.caption(f"Lý do: {dept_reason}")

        # Deep explanations (short in intake)
        st.markdown("### 🔎 Lý do nổi bật (AI)")
        st.write("• " + "\n• ".join(top_reasons(contrib_sorted)))

        # SBAR summary
        sbar = (
            f"SBAR: BN {p.age}t. GCS {gcs_total}/15. HR {p.hr}. SBP {p.sbp}. "
            f"RR {p.rr}. SpO2 {p.spo2}%. Temp {p.temp}. "
            f"EWS {ews}, SI {si}. "
            f"Risk {risk*100:.1f}%, Unc {u:.3f}. "
            f"Phân loại: {triage}. Chuyển khoa: {dept}."
        )
        st.text_area("Tóm tắt (SBAR):", sbar)

        # Save last_case + logs (audit)
        last_case = {
            "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Tuổi": p.age, "HR": p.hr, "SBP": p.sbp, "SpO2": p.spo2, "RR": p.rr, "Temp": p.temp,
            "GCS": gcs_total, "E": p.gcs_e, "V": p.gcs_v, "M": p.gcs_m,
            "Đau ngực": p.chest_pain, "Khó thở": p.dyspnea, "Chấn thương": p.trauma, "VAS": p.pain_level,
            "Khởi phát": p.onset, "Diễn tiến": p.progression,
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
        st.session_state["last_case"] = {
            "patient": asdict(p),
            "derived": {"gcs": gcs_total, "ews": ews, "si": si, "risk": risk, "u": u},
            "decision": {"triage": triage, "note": note, "flags": flags},
            "dept": {"name": dept, "reason": dept_reason},
            "explain": {
                "top_reasons": top_reasons(contrib_sorted),
                "uncertainty_reasons": uncertainty_reasons(p, ews, risk, u),
                "actions": action_suggestions(triage, dept),
                "contrib_table": [{"feature": k, "label": FEATURE_LABELS.get(k, k), "contribution": float(v)}
                                  for k, v in list(contrib_sorted.items())[:12]],
                "preds": preds,
            },
            "sbar": sbar,
            "time": last_case["Thời gian"]
        }
        st.session_state["logs"].append(last_case)

        st.markdown("<div class='small-note'>⚠️ Demo nghiên cứu/học thuật. Quyết định cuối cùng thuộc bác sĩ.</div>",
                    unsafe_allow_html=True)

# -------------------------
# TAB 2: DASHBOARD
# -------------------------
with tab2:
    if st.session_state["logs"]:
        df = pd.DataFrame(st.session_state["logs"])
        colA, colB = st.columns([1, 1])

        with colA:
            st.subheader("Tỷ lệ bệnh nhân theo phân màu")
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
# TAB 3: LOGS / EXPORT + DEEP EXPLAIN
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
            # Panel 1: Safety rules
            st.markdown("### 1) Luật an toàn (Hard rules)")
            flags = case["decision"]["flags"]
            if flags:
                st.error("Kích hoạt red flags: " + ", ".join(flags))
            else:
                st.success("Không kích hoạt red flags.")

            # Panel 2: Clinical scores
            st.markdown("### 2) Điểm lâm sàng (EWS / Shock Index / GCS)")
            d = case["derived"]
            st.write(f"- **GCS:** {d['gcs']}/15")
            st.write(f"- **EWS:** {d['ews']}")
            st.write(f"- **Shock Index:** {d['si']}")

            # Panel 3: AI risk
            st.markdown("### 3) AI Risk (tại sao ra % này?)")
            st.write(f"- **Risk:** {d['risk']*100:.1f}%")
            st.write("- **Lý do nổi bật:** " + "; ".join(case["explain"]["top_reasons"]))
            with st.expander("Bảng đóng góp đặc trưng (giải thích sâu)"):
                st.dataframe(pd.DataFrame(case["explain"]["contrib_table"]), use_container_width=True)

            # Panel 4: Uncertainty
            st.markdown("### 4) Uncertainty (vì sao chắc/không chắc?)")
            st.write(f"- **σ:** {d['u']:.3f} ({uncertainty_level(d['u'])})")
            st.write("• " + "\n• ".join(case["explain"]["uncertainty_reasons"]))
            with st.expander("Phân bố dự đoán của ensemble (debug)"):
                st.write(pd.DataFrame({"p_i": case["explain"]["preds"]}).describe())

            # Panel 5: Decision + routing
            st.markdown("### 5) Quyết định + Chuyển khoa")
            st.write(f"- **Triage:** {case['decision']['triage']}")
            st.write(f"- **Ghi chú:** {case['decision']['note']}")
            st.write(f"- **Khoa đề xuất:** {case['dept']['name']}")
            st.caption("Lý do: " + case["dept"]["reason"])

            # Panel 6: Actions
            st.markdown("### 6) Gợi ý hành động (Actionable)")
            st.write("• " + "\n• ".join(case["explain"]["actions"]))

            with st.expander("SBAR (để chuyển giao nhanh)"):
                st.text(case["sbar"])
