from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional: load trained model if you provide one (joblib)
try:
    import joblib
except Exception:
    joblib = None

# =========================
# Metadata
# =========================
APP_NAME = "TriageAI"
APP_VERSION = "3.6-hoathanhque"
MODEL_VERSION = "ensemble-v1 (demo weights) + optional trained model"
REPO_HINT = "Deploy from GitHub on Streamlit Cloud"

# =========================
# Safety & thresholds
# =========================
RISK_RED = 0.70
RISK_YELLOW = 0.30

UNC_MED = 0.10
UNC_HIGH = 0.20

MODEL_PATH = Path("models/trained_model.joblib")  # optional

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

# =========================
# Utils
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sigmoid(z: float) -> float:
    z = clamp(z, -40, 40)
    return 1.0 / (1.0 + math.exp(-z))

def avpu_idx(avpu: str) -> int:
    return {"A": 0, "V": 1, "P": 2, "U": 3}.get(avpu, 0)

# =========================
# 1) Input validation
# =========================
def validate_inputs(p: Patient) -> Tuple[bool, List[str]]:
    hard = []
    soft = []

    if not (0 <= p.age <= 120): hard.append("Age must be 0–120.")
    if not (30 <= p.hr <= 220): hard.append("HR must be 30–220 bpm.")
    if not (50 <= p.sbp <= 250): hard.append("SBP must be 50–250 mmHg.")
    if not (50 <= p.spo2 <= 100): hard.append("SpO₂ must be 50–100%.")
    if not (5 <= p.rr <= 60): hard.append("RR must be 5–60 /min.")
    if not (34.0 <= p.temp_c <= 42.0): hard.append("Temp must be 34–42°C.")
    if p.avpu not in {"A", "V", "P", "U"}: hard.append("AVPU must be A/V/P/U.")

    # soft consistency checks
    if p.spo2 < 88 and not p.severe_dyspnea:
        soft.append("SpO₂ very low but severe dyspnea not checked (re-check).")
    if p.sbp < 90 and p.hr < 60:
        soft.append("Low SBP with low HR can be unusual—verify measurements.")

    ok = len(hard) == 0
    issues = hard + soft
    return ok, issues

# =========================
# 2) Hard safety rules (Red flags)
# =========================
def red_flags(p: Patient) -> List[str]:
    flags = []
    if p.spo2 < 90: flags.append("SpO₂ < 90%")
    if p.sbp < 90: flags.append("SBP < 90 mmHg")
    if avpu_idx(p.avpu) >= 2: flags.append("AVPU = P/U (reduced consciousness)")
    if p.severe_dyspnea: flags.append("Severe dyspnea")
    if p.hr >= 140: flags.append("HR ≥ 140")
    if p.rr >= 30: flags.append("RR ≥ 30")
    return flags

# =========================
# 3) Feature engineering (interpretable)
# =========================
def features(p: Patient) -> Dict[str, float]:
    return {
        "age": float(p.age),
        "hr_excess": float(max(0, p.hr - 90)),
        "sbp_drop": float(max(0, 100 - p.sbp)),
        "spo2_drop": float(max(0, 95 - p.spo2)),
        "rr_excess": float(max(0, p.rr - 18)),
        "temp_excess": float(max(0, p.temp_c - 37.5)),
        "avpu": float(avpu_idx(p.avpu)),
        "chest_pain": float(1 if p.chest_pain else 0),
        "trauma": float(1 if p.trauma else 0),
        "severe_dyspnea": float(1 if p.severe_dyspnea else 0),
    }

FEATURE_LABELS = {
    "spo2_drop": "Low SpO₂",
    "sbp_drop": "Low SBP",
    "hr_excess": "High HR",
    "rr_excess": "High RR",
    "avpu": "Reduced consciousness",
    "temp_excess": "Fever",
    "chest_pain": "Chest pain",
    "trauma": "Trauma",
    "severe_dyspnea": "Severe dyspnea",
    "age": "Age",
}

# =========================
# 4) AI layer (two modes)
#    A) Optional trained model (if provided)
#    B) Otherwise: ensemble logistic (demo) with uncertainty = std
# =========================
@st.cache_resource(show_spinner=False)
def load_trained_model() -> Optional[object]:
    if joblib is None:
        return None
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def ensemble_params(seed: int = 42, n_models: int = 21) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    base = {
        "b0": -7.0,
        "age": 0.012,
        "hr_excess": 0.020,
        "sbp_drop": 0.050,
        "spo2_drop": 0.115,
        "rr_excess": 0.028,
        "temp_excess": 0.45,
        "avpu": 0.95,
        "chest_pain": 0.22,
        "trauma": 0.35,
        "severe_dyspnea": 0.55,
    }

    models = []
    for _ in range(n_models):
        m = {"b0": base["b0"] + rng.normal(0, 0.25)}
        for k in base:
            if k == "b0":
                continue
            m[k] = base[k] * (1.0 + rng.normal(0, 0.10))
        models.append(m)
    return models

@st.cache_data(show_spinner=False)
def get_ensemble() -> List[Dict[str, float]]:
    return ensemble_params(seed=42, n_models=21)

def predict_with_ensemble(p: Patient) -> Tuple[float, float, Dict[str, float], List[float]]:
    x = features(p)
    models = get_ensemble()

    ps = []
    for m in models:
        z = m["b0"]
        for k, v in x.items():
            z += m.get(k, 0.0) * v
        ps.append(sigmoid(z))

    arr = np.array(ps, dtype=float)
    mean_p = float(arr.mean())
    std_p = float(arr.std(ddof=1))

    # explanation: contribution using first model weights
    base = models[0]
    contrib = {k: float(base.get(k, 0.0) * v) for k, v in x.items()}
    contrib = dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return mean_p, std_p, contrib, ps

def predict_with_trained_model(p: Patient, model) -> Tuple[float, float, Dict[str, float], List[float]]:
    """
    If you train a model, replace this with your real pipeline.
    For now:
    - If model supports predict_proba: use it.
    - Uncertainty fallback: 0.0 (unless you implement ensemble for trained model).
    """
    x = features(p)
    X = pd.DataFrame([x])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        proba = float(model.predict(X)[0])

    contrib = {k: 0.0 for k in x.keys()}  # placeholder for real explainers
    return proba, 0.0, contrib, [proba]

# =========================
# 5) Conformal (optional, lightweight)
#    If uncertainty is high or risk in gray zone, output a SET of labels.
# =========================
def conformal_label_set(risk: float, u: float) -> List[str]:
    """
    Simple conformal-style behavior (demo):
    - High uncertainty => return multiple possible labels
    - Otherwise => single label
    """
    if u >= UNC_HIGH:
        # very uncertain -> wide set
        return ["🟢 GREEN", "🟡 YELLOW", "🔴 RED"]
    if u >= UNC_MED:
        # medium uncertain -> likely 2 labels
        if risk >= RISK_RED:
            return ["🟡 YELLOW", "🔴 RED"]
        if risk >= RISK_YELLOW:
            return ["🟢 GREEN", "🟡 YELLOW"]
        return ["🟢 GREEN", "🟡 YELLOW"]
    return [triage_label(risk)]

# =========================
# 6) Decision policy (Human-in-the-loop)
# =========================
def triage_label(risk: float) -> str:
    if risk >= RISK_RED:
        return "🔴 RED"
    if risk >= RISK_YELLOW:
        return "🟡 YELLOW"
    return "🟢 GREEN"

def uncertainty_level(u: float) -> str:
    if u >= UNC_HIGH:
        return "HIGH"
    if u >= UNC_MED:
        return "MEDIUM"
    return "LOW"

def decision_policy(flags: List[str], risk: float, u: float) -> Tuple[str, str]:
    # Hard safety
    if flags:
        return "🔴 RED (Hard safety)", "Red flags triggered: " + "; ".join(flags)

    # Soft safety / HITL
    ul = uncertainty_level(u)
    base = triage_label(risk)

    if ul == "HIGH":
        return base + " (Review required)", "High uncertainty: do NOT auto-triage. Clinician review required."
    if ul == "MEDIUM":
        return base + " (Confirm)", "Medium uncertainty: re-check vitals / ask key questions before finalizing."
    return base, "Low uncertainty: model agreement is high (still clinician decides)."

def top_reasons(contrib: Dict[str, float], k: int = 5) -> str:
    picks = []
    for feat, val in list(contrib.items())[:k]:
        if abs(val) < 0.05:
            continue
        picks.append(FEATURE_LABELS.get(feat, feat))
    return "; ".join(picks) if picks else "No dominant factors."

# =========================
# 7) Logging / Audit
# =========================
def append_log(p: Patient, risk: float, u: float, suggestion: str, note: str, flags: List[str], reasons: str, label_set: List[str]):
    st.session_state.setdefault("logs", [])
    st.session_state["logs"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **asdict(p),
        "risk": risk,
        "uncertainty": u,
        "uncertainty_level": uncertainty_level(u),
        "suggestion": suggestion,
        "note": note,
        "red_flags": "; ".join(flags),
        "reasons": reasons,
        "label_set": " | ".join(label_set),
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
    })

# =========================
# UI
# =========================
st.title(f"{APP_NAME} – High‑Reliability Triage Support")
st.caption("Safety-first: Red flags → always override. Risk + Uncertainty + (optional) Conformal label set + Audit logs.")

with st.sidebar:
    st.subheader("System Info")
    st.write(f"**App:** {APP_VERSION}")
    st.write(f"**Model:** {MODEL_VERSION}")
    st.write(f"**Deploy:** {REPO_HINT}")
    st.markdown("---")
    use_conformal = st.toggle("Use conformal label set (recommended for safety demos)", value=True)
    st.caption("⚠️ Research/demo only. Not for clinical decisions without validation.")

tab_eval, tab_method, tab_logs = st.tabs(["🧾 Evaluate", "📘 Method & Safety", "📤 Logs / Export"])

# ---- Evaluate
with tab_eval:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Vitals")
        age = st.number_input("Age", 0, 120, 40)
        hr = st.number_input("HR (bpm)", 30, 220, 90)
        sbp = st.number_input("SBP (mmHg)", 50, 250, 120)
        spo2 = st.number_input("SpO₂ (%)", 50, 100, 98)
        rr = st.number_input("RR (/min)", 5, 60, 18)
        temp_c = st.number_input("Temp (°C)", 34.0, 42.0, 37.0, 0.1)

    with c2:
        st.subheader("Status")
        avpu = st.selectbox("AVPU", ["A", "V", "P", "U"], index=0, help="A: alert, V: voice, P: pain, U: unresponsive")
        severe_dyspnea = st.checkbox("Severe dyspnea")
        chest_pain = st.checkbox("Chest pain")

    with c3:
        st.subheader("Context")
        trauma = st.checkbox("Trauma")
        st.markdown("### Thresholds")
        st.write(f"- RED if risk ≥ {int(RISK_RED*100)}%")
        st.write(f"- YELLOW if {int(RISK_YELLOW*100)}–{int(RISK_RED*100)-1}%")
        st.write(f"- GREEN if < {int(RISK_YELLOW*100)}%")

    patient = Patient(
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
    )

    ok, issues = validate_inputs(patient)
    if issues:
        st.warning("Input checks:\n- " + "\n- ".join(issues))

    flags = red_flags(patient)
    if flags:
        st.error("Hard safety red flags detected:\n- " + "\n- ".join(flags))

    run = st.button("Run assessment", type="primary", use_container_width=True, disabled=not ok)

    if run:
        trained = load_trained_model()

        try:
            if trained is not None:
                risk, u, contrib, ps = predict_with_trained_model(patient, trained)
            else:
                risk, u, contrib, ps = predict_with_ensemble(patient)

            suggestion, note = decision_policy(flags, risk, u)
            reasons = top_reasons(contrib)

            m1, m2, m3 = st.columns(3)
            m1.metric("Risk (P critical)", f"{risk*100:.1f}%")
            m2.metric("Uncertainty (σ)", f"{u:.3f}")
            m3.metric("Uncertainty level", uncertainty_level(u))

            if "🔴" in suggestion:
                st.error(f"**Suggestion:** {suggestion}\n\n{note}")
            elif "🟡" in suggestion:
                st.warning(f"**Suggestion:** {suggestion}\n\n{note}")
            else:
                st.success(f"**Suggestion:** {suggestion}\n\n{note}")

            st.markdown("### Explanation (for judges)")
            st.write(f"**Top reasons:** {reasons}")

            with st.expander("Debug: model disagreement distribution"):
                st.write(pd.DataFrame({"p_i": ps}).describe())

            label_set = conformal_label_set(risk, u) if use_conformal else [triage_label(risk)]
            st.markdown("### Safety output")
            st.write("**Label set (when uncertain):** " + " / ".join(label_set))

            append_log(patient, risk, u, suggestion, note, flags, reasons, label_set)

        except Exception as e:
            st.error("Model error → Fail-safe activated (rule-based only).")
            if flags:
                st.error("🔴 RED (Hard safety): " + "; ".join(flags))
            else:
                st.warning("🟡 YELLOW (Fail-safe): clinician review.")
            st.caption(f"Debug: {e!r}")

# ---- Method & Safety
with tab_method:
    st.subheader("Method & Safety")
    st.markdown(
        """
**High-reliability design (Safety-first)**  
1) **Input validation** prevents garbage-in.  
2) **Hard red flags** override all AI output.  
3) AI provides **Risk** (probability) + **Uncertainty** (model disagreement).  
4) **Human-in-the-loop policy**: uncertainty high → “review required”.  
5) **Conformal label set (optional)**: when uncertain, output multiple acceptable labels instead of a single risky claim.  
6) **Audit logs** store inputs/outputs with versioning.

**Why this is safer than “just accuracy”:**  
In triage, we optimize **miss rate of critical cases** and make the system **honest about uncertainty**.
        """
    )
    st.markdown("### What to say to judges (1 sentence)")
    st.info("We prioritize safety: red flags override AI, and uncertainty triggers clinician review instead of confident guesses.")

# ---- Logs / Export
with tab_logs:
    st.subheader("Audit logs")
    logs = st.session_state.get("logs", [])
    if not logs:
        st.info("No logs yet. Run a few cases in Evaluate.")
    else:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, height=360)

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="triageai_audit_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
