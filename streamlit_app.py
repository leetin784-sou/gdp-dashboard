# streamlit_app.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# App metadata
# =========================
APP_NAME = "TriageAI"
APP_VERSION = "1.0.0-safety"
MODEL_VERSION = "ensemble-logit-v1"
st.set_page_config(page_title=f"{APP_NAME} ({APP_VERSION})", layout="wide")

# =========================
# Data model
# =========================
@dataclass
class Patient:
    age: int
    hr: int
    sbp: int
    spo2: int
    rr: int
    temp_c: float
    avpu: str  # A/V/P/U
    chest_pain: bool
    trauma: bool
    severe_dyspnea: bool
    altered_mental: bool  # quick checkbox if not using AVPU deeply

@dataclass
class EvalResult:
    timestamp: str
    risk: float
    uncertainty: float
    uncertainty_level: str
    suggestion: str
    safety_note: str
    red_flags: str
    reasons: str

# =========================
# Utility
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(z: float) -> float:
    z = clamp(z, -40, 40)
    return 1.0 / (1.0 + math.exp(-z))

def avpu_to_idx(avpu: str) -> int:
    return {"A": 0, "V": 1, "P": 2, "U": 3}.get(avpu, 0)

# =========================
# 1) Input validation (reliability layer)
# =========================
def validate_inputs(p: Patient) -> Tuple[bool, List[str]]:
    issues = []

    # hard plausible ranges
    if not (0 <= p.age <= 120): issues.append("Tuổi ngoài phạm vi 0–120.")
    if not (30 <= p.hr <= 220): issues.append("HR ngoài phạm vi 30–220 bpm.")
    if not (50 <= p.sbp <= 250): issues.append("SBP ngoài phạm vi 50–250 mmHg.")
    if not (50 <= p.spo2 <= 100): issues.append("SpO₂ ngoài phạm vi 50–100%.")
    if not (5 <= p.rr <= 60): issues.append("RR ngoài phạm vi 5–60 /phút.")
    if not (34.0 <= p.temp_c <= 42.0): issues.append("Nhiệt độ ngoài phạm vi 34–42°C.")
    if p.avpu not in {"A", "V", "P", "U"}: issues.append("AVPU không hợp lệ.")

    # soft consistency checks (không khóa, chỉ cảnh báo)
    if p.spo2 < 85 and not p.severe_dyspnea:
        issues.append("SpO₂ rất thấp nhưng chưa tick 'khó thở nặng' (kiểm tra lại).")

    ok = len([x for x in issues if "ngoài phạm vi" in x or "không hợp lệ" in x]) == 0
    return ok, issues

# =========================
# 2) Safety rules (hard layer)
# =========================
def red_flags(p: Patient) -> List[str]:
    flags = []
    if p.spo2 < 90: flags.append("SpO₂ < 90%")
    if p.sbp < 90: flags.append("SBP < 90 mmHg")
    if avpu_to_idx(p.avpu) >= 2: flags.append("AVPU = P/U")
    if p.altered_mental: flags.append("Rối loạn tri giác (checkbox)")
    if p.severe_dyspnea: flags.append("Khó thở nặng")
    if p.hr >= 140: flags.append("HR ≥ 140")
    if p.rr >= 30: flags.append("RR ≥ 30")
    return flags

# =========================
# 3) Risk model (ensemble) + uncertainty
#    - “Tin cậy” hơn demo 1 model: nhiều model (ensemble)
#    - u = std(p_i) giữa các model
# =========================
def _features(p: Patient) -> Dict[str, float]:
    # features (giải thích được)
    return {
        "age": float(p.age),
        "hr_excess": float(max(0, p.hr - 90)),
        "sbp_drop": float(max(0, 100 - p.sbp)),
        "spo2_drop": float(max(0, 95 - p.spo2)),
        "rr_excess": float(max(0, p.rr - 18)),
        "temp_excess": float(max(0, p.temp_c - 37.5)),
        "avpu": float(avpu_to_idx(p.avpu)),
        "chest_pain": float(1 if p.chest_pain else 0),
        "trauma": float(1 if p.trauma else 0),
        "severe_dyspnea": float(1 if p.severe_dyspnea else 0),
        "altered_mental": float(1 if p.altered_mental else 0),
    }

def _ensemble_params(seed: int = 42) -> List[Dict[str, float]]:
    """
    Tạo 1 ensemble logistic n_models bộ beta hơi khác nhau.
    (Trong bài thật: bạn thay bằng model train từ data.)
    """
    rng = np.random.default_rng(seed)
    base = {
        "b0": -7.2,
        "age": 0.012,
        "hr_excess": 0.020,
        "sbp_drop": 0.050,
        "spo2_drop": 0.110,
        "rr_excess": 0.028,
        "temp_excess": 0.45,
        "avpu": 0.95,
        "chest_pain": 0.25,
        "trauma": 0.35,
        "severe_dyspnea": 0.55,
        "altered_mental": 0.60,
    }

    models = []
    for _ in range(15):
        m = {"b0": base["b0"] + rng.normal(0, 0.25)}
        for k in base:
            if k == "b0": 
                continue
            # jitter nhỏ giúp thể hiện uncertainty theo model disagreement
            m[k] = base[k] * (1.0 + rng.normal(0, 0.08))
        models.append(m)
    return models

@st.cache_data(show_spinner=False)
def get_models() -> List[Dict[str, float]]:
    return _ensemble_params(seed=42)

def predict_risk_and_uncertainty(p: Patient) -> Tuple[float, float, List[float], Dict[str, float]]:
    x = _features(p)
    models = get_models()

    ps = []
    for m in models:
        z = m["b0"]
        for k, v in x.items():
            z += m.get(k, 0.0) * v
        ps.append(sigmoid(z))

    ps_arr = np.array(ps, dtype=float)
    mean_p = float(ps_arr.mean())
    std_p = float(ps_arr.std(ddof=1))

    # “explanation”: contribution theo base weights (không phải SHAP nhưng giải thích được)
    base = models[0]
    contrib = {k: base.get(k, 0.0) * v for k, v in x.items()}
    # sort by absolute impact
    contrib_sorted = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return mean_p, std_p, ps, contrib_sorted

def uncertainty_level(u: float) -> str:
    if u >= 0.20: return "CAO"
    if u >= 0.10: return "TRUNG BÌNH"
    return "THẤP"

# =========================
# 4) Decision policy (Human-in-the-loop gate)
# =========================
def triage_from_risk(r: float) -> str:
    if r >= 0.70: return "🔴 ĐỎ"
    if r >= 0.30: return "🟡 VÀNG"
    return "🟢 XANH"

def decision(p: Patient, mean_risk: float, u: float, flags: List[str]) -> Tuple[str, str]:
    """
    Safety-first:
    - Red flags => ĐỎ ngay
    - Nếu không red flags:
        + risk cao nhưng u cao => yêu cầu bác sĩ đánh giá kỹ
        + u thấp => có thể khuyến nghị mạnh hơn
    """
    if flags:
        return "🔴 ĐỎ (Red flags)", "Luật an toàn kích hoạt: " + "; ".join(flags)

    base = triage_from_risk(mean_risk)
    ul = uncertainty_level(u)

    if ul == "CAO":
        note = "⚠️ Uncertainty CAO → không đưa khuyến nghị mạnh; cần bác sĩ đánh giá kỹ."
    elif ul == "TRUNG BÌNH":
        note = "Uncertainty TRUNG BÌNH → khuyến nghị kiểm tra thêm (đo lại vitals/khai thác)."
    else:
        note = "Uncertainty THẤP → mô hình khá tự tin."

    # Nếu risk rất cao mà u thấp → cảnh báo mạnh
    if mean_risk >= 0.80 and ul == "THẤP":
        note = "⚠️ Cảnh báo mạnh: Risk rất cao & Uncertainty thấp → ưu tiên xử trí ngay."
    return base, note

def format_reasons(contrib_sorted: Dict[str, float], topk: int = 5) -> str:
    # map keys to human text
    name = {
        "spo2_drop": "SpO₂ thấp",
        "sbp_drop": "Huyết áp thấp",
        "hr_excess": "Mạch nhanh",
        "rr_excess": "Thở nhanh",
        "avpu": "Tri giác giảm",
        "temp_excess": "Sốt",
        "chest_pain": "Đau ngực",
        "trauma": "Chấn thương",
        "severe_dyspnea": "Khó thở nặng",
        "altered_mental": "Rối loạn tri giác",
        "age": "Tuổi",
    }
    items = []
    for k, v in list(contrib_sorted.items())[:topk]:
        if abs(v) < 0.05:
            continue
        items.append(name.get(k, k))
    return "; ".join(items) if items else "Không có yếu tố nổi bật"

# =========================
# UI
# =========================
st.title(f"{APP_NAME} – Risk + Uncertainty")
st.caption("Bản Safety‑first: Rule‑based (red flags) + Risk score + Uncertainty + Human‑in‑the‑loop + Logging.")
with st.sidebar:
    st.subheader("Thông tin hệ thống")
    st.write(f"- App version: **{APP_VERSION}**")
    st.write(f"- Model version: **{MODEL_VERSION}**")
    st.markdown("---")
    st.caption("⚠️ Demo phục vụ học thuật/thuyết trình. Không dùng cho quyết định lâm sàng thật.")

tabs = st.tabs(["🧾 Đánh giá", "📊 Method & Safety", "📤 Logs/Export"])

# ---------- Tab 1: Evaluate ----------
with tabs[0]:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Vitals")
        age = st.number_input("Tuổi", 0, 120, 40)
        hr = st.number_input("Mạch (HR, bpm)", 30, 220, 90)
        sbp = st.number_input("Huyết áp tâm thu (SBP, mmHg)", 50, 250, 120)
        spo2 = st.number_input("SpO₂ (%)", 50, 100, 98)
        rr = st.number_input("Nhịp thở (RR, /phút)", 5, 60, 18)
        temp_c = st.number_input("Nhiệt độ (°C)", 34.0, 42.0, 37.0, 0.1)

    with col2:
        st.subheader("Tình trạng")
        avpu = st.selectbox("AVPU", ["A", "V", "P", "U"], index=0, help="A: tỉnh; V: đáp ứng lời; P: đáp ứng đau; U: không đáp ứng")
        altered_mental = st.checkbox("Rối loạn tri giác (nếu có)")
        severe_dyspnea = st.checkbox("Khó thở nặng")

    with col3:
        st.subheader("Bối cảnh")
        chest_pain = st.checkbox("Đau ngực")
        trauma = st.checkbox("Chấn thương")

        st.markdown("### Ngưỡng gợi ý (demo)")
        st.write("- Risk ≥ 0.70 → Đỏ")
        st.write("- 0.30–0.69 → Vàng")
        st.write("- < 0.30 → Xanh")

    p = Patient(
        age=int(age),
        hr=int(hr),
        sbp=int(sbp),
        spo2=int(spo2),
        rr=int(rr),
        temp_c=float(temp_c),
        avpu=str(avpu),
        chest_pain=bool(chest_pain),
        trauma=bool(trauma),
        severe_dyspnea=bool(severe_dyspnea),
        altered_mental=bool(altered_mental),
    )

    ok, issues = validate_inputs(p)
    if issues:
        st.warning("**Kiểm tra dữ liệu:**\n- " + "\n- ".join(issues))

    # button disabled if hard-invalid
    run = st.button("Đánh giá Risk + Uncertainty", type="primary", use_container_width=True, disabled=not ok)

    if run:
        try:
            flags = red_flags(p)
            mean_risk, u, ps, contrib_sorted = predict_risk_and_uncertainty(p)
            suggestion, safety_note = decision(p, mean_risk, u, flags)
            reasons = format_reasons(contrib_sorted)

            a, b, c = st.columns(3)
            a.metric("Risk score (P nguy kịch)", f"{mean_risk*100:.1f}%")
            b.metric("Uncertainty (σ)", f"{u:.3f}")
            c.metric("Mức tin cậy", uncertainty_level(u))

            if "🔴" in suggestion:
                st.error(f"**Gợi ý phân luồng:** {suggestion}\n\n{safety_note}")
            elif "🟡" in suggestion:
                st.warning(f"**Gợi ý phân luồng:** {suggestion}\n\n{safety_note}")
            else:
                st.success(f"**Gợi ý phân luồng:** {suggestion}\n\n{safety_note}")

            st.markdown("### Giải thích (để thuyết trình)")
            st.write(f"**Lý do nổi bật:** {reasons}")
            with st.expander("Xem đóng góp đặc trưng (debug/giải thích sâu)"):
                dfc = pd.DataFrame(
                    [{"feature": k, "contribution": float(v)} for k, v in list(contrib_sorted.items())[:10]]
                )
                st.dataframe(dfc, use_container_width=True)

            # Save log
            ts = datetime.now().isoformat(timespec="seconds")
            res = EvalResult(
                timestamp=ts,
                risk=mean_risk,
                uncertainty=u,
                uncertainty_level=uncertainty_level(u),
                suggestion=suggestion,
                safety_note=safety_note,
                red_flags="; ".join(flags) if flags else "",
                reasons=reasons,
            )
            log_row = {**asdict(p), **asdict(res)}
            st.session_state.setdefault("logs", [])
            st.session_state["logs"].append(log_row)

        except Exception as e:
            # fail-safe fallback
            st.error("Có lỗi khi tính AI. Hệ thống chuyển sang chế độ an toàn (rule-based).")
            flags = red_flags(p)
            if flags:
                st.error("🔴 ĐỎ (Red flags) – " + "; ".join(flags))
            else:
                st.warning("🟡 VÀNG (Fallback) – Khuyến nghị bác sĩ đánh giá lâm sàng.")
            st.caption(f"Debug (không cần đưa vào báo cáo): {e!r}")

# ---------- Tab 2: Method & Safety ----------
with tabs[1]:
    st.subheader("Method & Safety (để BGK đọc)")
    st.markdown(
        """
**Kiến trúc tin cậy cao (Safety-first):**
1) **Input validation**: dữ liệu ngoài phạm vi hợp lý → không cho đánh giá (giảm “rác vào”).  
2) **Hard rules / Red flags**: kích hoạt ưu tiên **ĐỎ** ngay (an toàn là trên hết).  
3) **Risk score**: ước lượng xác suất nguy kịch **liên tục** (0–100%).  
4) **Uncertainty**: tính từ **ensemble** (mức bất đồng giữa nhiều mô hình) → biết khi nào “không chắc”.  
5) **Human-in-the-loop**: Uncertainty cao → **không áp đặt**, yêu cầu bác sĩ đánh giá kỹ.  
6) **Logging/Audit**: lưu input–output để truy vết, xuất CSV.

**Vì sao không nói “đúng 99%”?**  
Đề tài y tế ưu tiên **an toàn**: đo **Recall lớp nguy kịch** và **tỷ lệ bỏ sót nguy kịch**, hơn là accuracy chung.
        """
    )
    st.markdown("### Quy tắc quyết định (tóm tắt)")
    st.code(
        "Nếu red flags → ĐỎ ngay\n"
        "Nếu không: dùng Risk + Uncertainty\n"
        "- Risk cao & Uncertainty thấp → cảnh báo mạnh\n"
        "- Uncertainty cao → yêu cầu bác sĩ đánh giá",
        language="text",
    )

# ---------- Tab 3: Logs / Export ----------
with tabs[2]:
    st.subheader("Logs / Export (Audit trail)")
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("Chưa có log. Hãy chạy vài ca ở tab **Đánh giá**.")
    else:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, height=350)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tải logs CSV",
            data=csv,
            file_name="triageai_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
