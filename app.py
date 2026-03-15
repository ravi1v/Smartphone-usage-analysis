import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartSense — Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #0a0a0f;
    background-image:
        linear-gradient(rgba(124,58,237,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}

section[data-testid="stSidebar"] {
    background: #12121a !important;
    border-right: 1px solid #2a2a3e;
}
section[data-testid="stSidebar"] * { color: #f0f0ff !important; }

.block-container { padding-top: 1.5rem; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #f0f0ff !important; }

[data-testid="metric-container"] {
    background: #1a1a26;
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #f0f0ff !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    color: #6b6b8a !important;
    text-transform: uppercase;
}

.stSelectbox > div > div {
    background: #12121a !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 9px !important;
    color: #f0f0ff !important;
    font-family: 'Space Mono', monospace !important;
}

.stNumberInput > div > div > input {
    background: #12121a !important;
    border: 1px solid #2a2a3e !important;
    color: #f0f0ff !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 9px !important;
}

.stRadio > div { flex-direction: row !important; gap: 12px; }
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important;
}

hr { border-color: #2a2a3e !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #12121a;
    border-radius: 10px;
    gap: 4px;
    border-bottom: 1px solid #2a2a3e;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px;
    color: #6b6b8a !important;
    background: transparent !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,0.2) !important;
    color: #a78bfa !important;
}

label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.5px !important;
    color: #6b6b8a !important;
    text-transform: uppercase !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL  (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "addiction_pipeline.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()


# ─────────────────────────────────────────────
# FEATURE LIST & ENGINEERING  (mirrors notebook)
# ─────────────────────────────────────────────
FEATURES = [
    'age', 'gender', 'daily_screen_time_hours', 'social_media_hours',
    'gaming_hours', 'work_study_hours', 'sleep_hours',
    'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time',
    'stress_level', 'academic_work_impact',
    'social_media_screen_ratio', 'gaming_screen_ratio',
    'pickups_per_hour', 'sleep_deprivation_risk', 'high_social_media'
]

def engineer_features(data: dict) -> dict:
    d = data.copy()
    d["social_media_screen_ratio"] = d["social_media_hours"] / (d["daily_screen_time_hours"] + 0.01)
    d["gaming_screen_ratio"]       = d["gaming_hours"]       / (d["daily_screen_time_hours"] + 0.01)
    d["pickups_per_hour"]          = d["app_opens_per_day"]  / 16
    d["sleep_deprivation_risk"]    = 1 if d["sleep_hours"] < 6 else 0
    d["high_social_media"]         = 1 if d["social_media_hours"] > 3 else 0
    return d

def build_feature_row(data: dict) -> pd.DataFrame:
    """Single-row DataFrame in the exact column order the model was trained on."""
    d = engineer_features(data)
    return pd.DataFrame([[d[f] for f in FEATURES]], columns=FEATURES)


# ─────────────────────────────────────────────
# PREDICTION
# Uses the real pkl model; falls back to a
# rule-based scorer if the file is missing.
# ─────────────────────────────────────────────
def predict_addiction(data: dict):
    """Returns (probability_pct: int, label: int, source: str)"""
    if model is not None:
        # Build feature dict with engineered columns, then DataFrame
        features = engineer_features(data)
        df = pd.DataFrame([features])          # single-row DataFrame
        df = df[FEATURES]                      # enforce exact column order
        prob  = round(float(model.predict_proba(df)[0][1]) * 100)
        label = int(model.predict(df)[0])
        return prob, label, "model"

    # ── Rule-based fallback ──
    d = engineer_features(data)
    s = 0.0
    s += min((d["daily_screen_time_hours"] / 16) * 28, 28)
    s += min((d["weekend_screen_time"]     / 18) * 14, 14)
    s += min((d["social_media_hours"]      / 12) * 12, 12)
    s += 6 if d["social_media_screen_ratio"] > 0.5 else (3 if d["social_media_screen_ratio"] > 0.3 else 0)
    s += 8 if d["sleep_deprivation_risk"] else (3 if d["sleep_hours"] < 7 else 0)
    s += min((d["app_opens_per_day"]       / 400) * 10, 10)
    s += min((d["notifications_per_day"]   / 500) *  6,  6)
    s += 5 if d["high_social_media"] else 0
    s += 5 if d["stress_level"] == 0 else (2 if d["stress_level"] == 2 else 0)
    s += 4 if d["academic_work_impact"] == 0 else 0
    s += 3 if d["age"] < 20 else (1.5 if d["age"] < 25 else 0)
    s += min((d["gaming_hours"] / 10) * 5, 5)
    prob = round(min(max(s, 0), 100))
    return prob, (1 if prob >= 50 else 0), "fallback"


# ─────────────────────────────────────────────
# COLOUR / LABEL HELPERS
# ─────────────────────────────────────────────
def risk_color(p: float) -> str:
    if p >= 75: return "#ef4444"
    if p >= 50: return "#f59e0b"
    if p >= 30: return "#06b6d4"
    return "#10b981"

def risk_label(p: float) -> str:
    if p >= 75: return "⚠️ HIGH RISK"
    if p >= 50: return "⚡ MODERATE RISK"
    if p >= 30: return "🔹 MILD RISK"
    return "✅ LOW RISK"

def hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
c1, c2, c3 = st.columns([0.07, 0.65, 0.28])
with c1:
    st.markdown("<div style='font-size:2.4rem;margin-top:6px'>📱</div>", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <h1 style='margin:0;font-size:1.9rem;font-weight:800;
    background:linear-gradient(135deg,#fff 40%,#06b6d4);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.2;'>
    SmartSense</h1>
    <p style='margin:0;font-family:Space Mono,monospace;font-size:0.68rem;
    color:#6b6b8a;letter-spacing:1.5px;'>SMARTPHONE ADDICTION RISK PREDICTOR</p>
    """, unsafe_allow_html=True)
with c3:
    if model is not None:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);
        border-radius:10px;padding:8px 14px;margin-top:8px;text-align:center;'>
          <div style='font-family:Space Mono,monospace;font-size:0.65rem;
          color:#10b981;font-weight:700;'>🟢  MODEL LOADED</div>
          <div style='font-family:Space Mono,monospace;font-size:0.6rem;
          color:#6b6b8a;margin-top:2px;'>smartphone_addiction_model.pkl</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
        border-radius:10px;padding:8px 14px;margin-top:8px;text-align:center;'>
          <div style='font-family:Space Mono,monospace;font-size:0.65rem;
          color:#f59e0b;font-weight:700;'>🟡  FALLBACK MODE</div>
          <div style='font-family:Space Mono,monospace;font-size:0.6rem;
          color:#6b6b8a;margin-top:2px;'>Rule-based scoring active</div>
        </div>""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ **`smartphone_addiction_model.pkl` not found.** Place it in the same folder as `app.py` to use your trained stacking ensemble. Currently running rule-based fallback scoring.")

st.markdown("<hr style='margin:16px 0 24px;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — INPUT CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#6b6b8a;
    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:20px;
    padding-bottom:12px;border-bottom:1px solid #2a2a3e;'>// Patient Profile Input</div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Demographics</div>", unsafe_allow_html=True)
    age    = st.number_input("Age",    min_value=10, max_value=70, value=22, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    gender_enc = {"Male": 0, "Female": 1, "Other": 2}[gender]

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Daily Screen Usage</div>", unsafe_allow_html=True)
    daily_screen   = st.slider("Daily Screen Time (hrs)",    0.0, 16.0,  6.0, 0.1)
    social_media   = st.slider("Social Media (hrs)",         0.0, 12.0,  2.5, 0.1)
    gaming         = st.slider("Gaming (hrs)",               0.0, 10.0,  1.0, 0.1)
    work_study     = st.slider("Work / Study (hrs)",         0.0, 12.0,  4.0, 0.1)
    sleep          = st.slider("Sleep (hrs)",                2.0, 12.0,  7.0, 0.1)
    weekend_screen = st.slider("Weekend Screen Time (hrs)",  0.0, 18.0,  8.0, 0.1)

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Behavioral Metrics</div>", unsafe_allow_html=True)
    notifications = st.number_input("Notifications / Day", min_value=0, max_value=500, value=120, step=5)
    app_opens     = st.number_input("App Opens / Day",     min_value=0, max_value=400, value=80,  step=5)

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#06b6d4;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>Psychological Indicators</div>", unsafe_allow_html=True)
    stress_raw = st.radio("Stress Level",           ["High", "Medium", "Low"], index=1, horizontal=True)
    acad_raw   = st.radio("Academic / Work Impact", ["Yes", "No"],             index=0, horizontal=True)
    # Encode exactly as training did
    stress_enc = {"High": 0, "Low": 1, "Medium": 2}[stress_raw]
    acad_enc   = 0 if acad_raw == "Yes" else 1

    st.markdown("---")
    predict_btn = st.button("⚡  Predict Addiction Risk")


# ─────────────────────────────────────────────
# COLLECT INPUT DICT
# ─────────────────────────────────────────────
input_data = {
    "age":                     age,
    "gender":                  gender_enc,
    "daily_screen_time_hours": daily_screen,
    "social_media_hours":      social_media,
    "gaming_hours":            gaming,
    "work_study_hours":        work_study,
    "sleep_hours":             sleep,
    "notifications_per_day":   notifications,
    "app_opens_per_day":       app_opens,
    "weekend_screen_time":     weekend_screen,
    "stress_level":            stress_enc,
    "academic_work_impact":    acad_enc,
}
feats = engineer_features(input_data)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# ─────────────────────────────────────────────
# RUN PREDICTION ON BUTTON CLICK
# ─────────────────────────────────────────────
if predict_btn:
    prob, label, source = predict_addiction(input_data)
    st.session_state.last_prediction = {
        "prob": prob, "label": label, "source": source,
        "feats": feats, "input": input_data.copy()
    }
    st.session_state.history.append({
        "prob": prob, "label": label, "source": source,
        "screen": daily_screen, "social": social_media, "sleep": sleep,
    })


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
pred = st.session_state.last_prediction

if pred is None:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;'>
      <div style='font-size:4rem;margin-bottom:20px;opacity:0.3;'>🔮</div>
      <h3 style='color:#6b6b8a;font-weight:600;margin-bottom:10px;'>No prediction yet</h3>
      <p style='color:#4a4a6a;font-family:Space Mono,monospace;font-size:0.8rem;'>
        Configure the patient profile in the sidebar<br>
        and click <strong style='color:#a78bfa;'>Predict Addiction Risk</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)

else:
    prob   = pred["prob"]
    label  = pred["label"]
    source = pred["source"]
    f      = pred["feats"]
    color  = risk_color(prob)

    # ── Always read display values from the SAVED prediction, not live sidebar ──
    _inp           = pred["input"]
    daily_screen   = _inp["daily_screen_time_hours"]
    social_media   = _inp["social_media_hours"]
    gaming         = _inp["gaming_hours"]
    work_study     = _inp["work_study_hours"]
    sleep          = _inp["sleep_hours"]
    notifications  = _inp["notifications_per_day"]
    app_opens      = _inp["app_opens_per_day"]
    weekend_screen = _inp["weekend_screen_time"]
    stress_enc     = _inp["stress_level"]
    acad_enc       = _inp["academic_work_impact"]
    age            = _inp["age"]

    # ── Source badge ──
    if source == "model":
        st.markdown("""
        <div style='display:inline-block;background:rgba(16,185,129,0.1);
        border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:5px 12px;
        font-family:Space Mono,monospace;font-size:0.65rem;color:#10b981;margin-bottom:16px;'>
        ✓ Prediction from <strong>smartphone_addiction_model.pkl</strong>
        (Stacking Ensemble · GBM + RF + ET → LogisticRegression)
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:inline-block;background:rgba(245,158,11,0.1);
        border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:5px 12px;
        font-family:Space Mono,monospace;font-size:0.65rem;color:#f59e0b;margin-bottom:16px;'>
        ⚡ Rule-based fallback — load model pkl for real predictions
        </div>""", unsafe_allow_html=True)

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Risk Score", f"{prob}%",
                  delta="Addicted" if label else "Not Addicted",
                  delta_color="inverse" if label else "normal")
    with m2:
        st.metric("Daily Screen", f"{daily_screen:.1f} hrs",
                  delta="High" if daily_screen > 8 else ("Moderate" if daily_screen > 5 else "OK"))
    with m3:
        st.metric("Sleep", f"{sleep:.1f} hrs",
                  delta="Deprived" if sleep < 6 else ("Good" if sleep >= 7 else "Low"),
                  delta_color="inverse" if sleep < 6 else "normal")
    with m4:
        st.metric("App Opens", f"{app_opens}",
                  delta="High" if app_opens > 150 else ("Moderate" if app_opens > 80 else "OK"))

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Risk Analysis",
        "🔍  Feature Breakdown",
        "💡  Recommendations",
        "📈  History"
    ])

    # ════════════════════════════════════
    # TAB 1 — RISK ANALYSIS
    # ════════════════════════════════════
    with tab1:
        col_gauge, col_radar = st.columns([0.42, 0.58])

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"family": "Space Mono", "size": 44, "color": color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2a2a3e",
                             "tickfont": {"family": "Space Mono", "size": 10, "color": "#6b6b8a"}},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "#1a1a26", "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30],  "color": "rgba(16,185,129,0.15)"},
                        {"range": [30, 50],  "color": "rgba(6,182,212,0.12)"},
                        {"range": [50, 75],  "color": "rgba(245,158,11,0.15)"},
                        {"range": [75, 100], "color": "rgba(239,68,68,0.18)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": prob},
                },
                title={"text": f"<b>{risk_label(prob)}</b>",
                       "font": {"family": "Syne", "size": 18, "color": color}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=60, b=10, l=30, r=30), height=300,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            verdict_map = {
                "⚠️ HIGH RISK":     "Strong indicators of smartphone addiction detected. Immediate intervention recommended.",
                "⚡ MODERATE RISK": "Several addiction patterns present. Professional consultation advised.",
                "🔹 MILD RISK":     "Minor risk factors present. Mindful usage habits recommended.",
                "✅ LOW RISK":      "Healthy smartphone usage patterns observed. Maintain current habits.",
            }
            st.markdown(f"""
            <div style='background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.2);
            border-radius:12px;padding:16px;margin-top:-10px;'>
              <div style='font-family:Space Mono,monospace;font-size:0.68rem;
              color:#6b6b8a;letter-spacing:1px;margin-bottom:8px;'>ASSESSMENT SUMMARY</div>
              <div style='font-size:0.85rem;color:#c4c4d8;line-height:1.6;'>
              {verdict_map[risk_label(prob)]}</div>
              <div style='margin-top:12px;font-family:Space Mono,monospace;font-size:0.68rem;'>
                <span style='color:#6b6b8a;'>Classification: </span>
                <span style='color:{color};font-weight:700;'>{"ADDICTED (1)" if label else "NOT ADDICTED (0)"}</span>
                &nbsp;&nbsp;
                <span style='color:#6b6b8a;'>Source: </span>
                <span style='color:#a78bfa;'>{"pkl model" if source=="model" else "fallback"}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_radar:
            categories = ["Screen Time","Social Media","Gaming",
                          "Sleep Health","Notifications","Weekend Usage","Work Balance"]
            raw_vals = [
                min(daily_screen / 16 * 100, 100),
                min(social_media / 12 * 100, 100),
                min(gaming       / 10 * 100, 100),
                max(0, 100 - sleep      / 12 * 100),
                min(notifications / 500 * 100, 100),
                min(weekend_screen / 18 * 100, 100),
                max(0, 100 - work_study / 12 * 100),
            ]
            vals = raw_vals + [raw_vals[0]]
            cats = categories + [categories[0]]
            rgb  = hex_rgb(color)

            fig_radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                fillcolor=f"rgba({rgb},0.15)",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,100],
                                   tickfont=dict(family="Space Mono", size=9, color="#6b6b8a"),
                                   gridcolor="#2a2a3e", linecolor="#2a2a3e"),
                    angularaxis=dict(tickfont=dict(family="Syne", size=11, color="#c4c4d8"),
                                     gridcolor="#2a2a3e", linecolor="#2a2a3e"),
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, margin=dict(t=30,b=30,l=50,r=50), height=340,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Risk factor cards
        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#6b6b8a;letter-spacing:1.5px;text-transform:uppercase;margin:8px 0 16px;'>// Risk Factor Analysis</div>", unsafe_allow_html=True)
        lc_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        factors = [
            ("📱 Daily Screen Time", daily_screen/16*100, f"{daily_screen:.1f} hrs/day",
             "High" if daily_screen>8 else "Medium" if daily_screen>5 else "Low"),
            ("💬 Social Media", social_media/12*100, f"{social_media:.1f} hrs · {f['social_media_screen_ratio']*100:.0f}% of screen",
             "High" if f["high_social_media"] else "Medium" if social_media>2 else "Low"),
            ("😴 Sleep Health", max(0,100-sleep/12*100), f"{sleep:.1f} hrs/night",
             "High" if f["sleep_deprivation_risk"] else "Medium" if sleep<7 else "Low"),
            ("🔔 Notifications", notifications/500*100, f"{notifications} notifs · {app_opens} opens/day",
             "High" if notifications>200 else "Medium" if notifications>100 else "Low"),
            ("🎮 Gaming", gaming/10*100, f"{gaming:.1f} hrs/day",
             "High" if gaming>4 else "Medium" if gaming>2 else "Low"),
            ("📅 Weekend Screen", weekend_screen/18*100, f"{weekend_screen:.1f} hrs",
             "High" if weekend_screen>12 else "Medium" if weekend_screen>8 else "Low"),
        ]
        fc1, fc2 = st.columns(2)
        for i, (name, pct, desc, lvl) in enumerate(factors):
            lc = lc_map[lvl]; r = hex_rgb(lc)
            with (fc1 if i%2==0 else fc2):
                st.markdown(f"""
                <div style='background:#1a1a26;border:1px solid #2a2a3e;border-radius:12px;
                padding:14px;margin-bottom:12px;'>
                  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                    <span style='font-weight:600;font-size:0.85rem;color:#f0f0ff;'>{name}</span>
                    <span style='font-family:Space Mono,monospace;font-size:0.62rem;padding:3px 9px;
                    border-radius:20px;background:rgba({r},0.15);color:{lc};
                    border:1px solid {lc}44;font-weight:700;'>{lvl}</span>
                  </div>
                  <div style='height:5px;background:#2a2a3e;border-radius:3px;margin-bottom:8px;overflow:hidden;'>
                    <div style='height:100%;width:{min(pct,100):.1f}%;
                    background:linear-gradient(90deg,{lc},{lc}aa);border-radius:3px;'></div>
                  </div>
                  <div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#6b6b8a;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════
    # TAB 2 — FEATURE BREAKDOWN
    # ════════════════════════════════════
    with tab2:
        fi_df = pd.DataFrame({
            "Feature": ["Daily Screen Time","Weekend Screen Time","Social Media Hours",
                        "App Opens / Day","Sleep Hours","Notifications / Day",
                        "Social Media Ratio","Gaming Hours","Stress Level","Pickups / Hour"],
            "Importance (%)": [22, 18, 15, 11, 9, 7, 6, 5, 4, 3],
        }).sort_values("Importance (%)")

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance (%)"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=fi_df["Importance (%)"],
                        colorscale=[[0,"#5b21b6"],[1,"#06b6d4"]], showscale=False),
            text=[f"{v}%" for v in fi_df["Importance (%)"]],
            textposition="outside",
            textfont=dict(family="Space Mono", size=11, color="#c4c4d8"),
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(tickfont=dict(family="Syne", size=12, color="#c4c4d8"), gridcolor="#2a2a3e"),
            margin=dict(t=40,b=10,l=10,r=60), height=360,
            title=dict(text="Feature Importance — Stacking Ensemble",
                       font=dict(family="Syne",size=14,color="#f0f0ff"), x=0),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#6b6b8a;letter-spacing:1.5px;text-transform:uppercase;margin:8px 0 14px;'>// Engineered Features (current values)</div>", unsafe_allow_html=True)
        ef1, ef2, ef3 = st.columns(3)
        eng = [
            ("Social Media Ratio",  f"{f['social_media_screen_ratio']:.3f}", "social_media / screen_time",   "#7c3aed"),
            ("Gaming Screen Ratio", f"{f['gaming_screen_ratio']:.3f}",       "gaming / screen_time",          "#06b6d4"),
            ("Pickups / Hour",      f"{f['pickups_per_hour']:.2f}",          "app_opens / 16",                "#10b981"),
            ("Sleep Deprivation",   "Yes" if f["sleep_deprivation_risk"] else "No", "sleep_hours < 6",
             "#ef4444" if f["sleep_deprivation_risk"] else "#10b981"),
            ("High Social Media",   "Yes" if f["high_social_media"] else "No", "social_media > 3 hrs",
             "#f59e0b" if f["high_social_media"] else "#10b981"),
        ]
        for i, (nm, val, formula, c) in enumerate(eng):
            with [ef1, ef2, ef3][i % 3]:
                st.markdown(f"""
                <div style='background:#1a1a26;border:1px solid #2a2a3e;border-radius:12px;
                padding:14px;margin-bottom:10px;'>
                  <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6b6b8a;
                  letter-spacing:1px;margin-bottom:6px;'>{nm}</div>
                  <div style='font-size:1.3rem;font-weight:800;color:{c};
                  font-family:Space Mono,monospace;'>{val}</div>
                  <div style='font-family:Space Mono,monospace;font-size:0.62rem;
                  color:#4a4a6a;margin-top:6px;'>= {formula}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#6b6b8a;letter-spacing:1.5px;text-transform:uppercase;margin:8px 0 14px;'>// Model Architecture</div>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("""
            <div style='background:#1a1a26;border:1px solid #2a2a3e;border-radius:12px;padding:18px;'>
              <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6b6b8a;
              letter-spacing:1px;margin-bottom:12px;'>BASE LEARNERS</div>
              <div style='display:flex;flex-direction:column;gap:8px;'>
                <div style='padding:8px 12px;background:#12121a;border-radius:8px;
                display:flex;justify-content:space-between;'>
                  <span style='font-size:0.82rem;color:#f0f0ff;'>GradientBoostingClassifier</span>
                  <span style='font-family:Space Mono,monospace;font-size:0.65rem;color:#a78bfa;'>n=200 · lr=0.05</span>
                </div>
                <div style='padding:8px 12px;background:#12121a;border-radius:8px;
                display:flex;justify-content:space-between;'>
                  <span style='font-size:0.82rem;color:#f0f0ff;'>RandomForestClassifier</span>
                  <span style='font-family:Space Mono,monospace;font-size:0.65rem;color:#a78bfa;'>n=300 · depth=10</span>
                </div>
                <div style='padding:8px 12px;background:#12121a;border-radius:8px;
                display:flex;justify-content:space-between;'>
                  <span style='font-size:0.82rem;color:#f0f0ff;'>ExtraTreesClassifier</span>
                  <span style='font-family:Space Mono,monospace;font-size:0.65rem;color:#a78bfa;'>n=200 · depth=12</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with a2:
            st.markdown("""
            <div style='background:#1a1a26;border:1px solid #2a2a3e;border-radius:12px;padding:18px;'>
              <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6b6b8a;
              letter-spacing:1px;margin-bottom:12px;'>PIPELINE</div>
              <div style='display:flex;flex-direction:column;gap:6px;
              font-family:Space Mono,monospace;font-size:0.73rem;line-height:1.8;'>
                <div style='color:#c4c4d8;'>① 12 raw input features</div>
                <div style='color:#6b6b8a;padding-left:10px;'>↓  +5 engineered features (17 total)</div>
                <div style='color:#c4c4d8;'>② StandardScaler normalisation</div>
                <div style='color:#6b6b8a;padding-left:10px;'>↓  80/20 stratified split</div>
                <div style='color:#c4c4d8;'>③ StackingClassifier (cv=5)</div>
                <div style='color:#6b6b8a;padding-left:10px;'>↓  Meta: LogisticRegression</div>
                <div style='color:#a78bfa;font-weight:700;'>④ predict() / predict_proba()</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════
    # TAB 3 — RECOMMENDATIONS
    # ════════════════════════════════════
    with tab3:
        recs = []
        if daily_screen > 6:
            recs.append(("📱","Reduce Screen Time",
                f"You average {daily_screen:.1f} hrs/day. Set a cap of 4–5 hours using Digital Wellbeing / Screen Time tools.",
                "#ef4444" if daily_screen>10 else "#f59e0b"))
        if social_media > 3:
            recs.append(("💬","Limit Social Media",
                f"{social_media:.1f} hrs/day on social media is excessive. Use app timers and try grayscale mode.","#ef4444"))
        if f["sleep_deprivation_risk"]:
            recs.append(("😴","Improve Sleep Hygiene",
                f"Only {sleep:.1f} hrs sleep — below the 7–9 hr recommendation. No phone 1 hr before bed.","#ef4444"))
        elif sleep < 7:
            recs.append(("😴","Boost Sleep Duration",
                f"{sleep:.1f} hrs is slightly low. Aim for 7–8 hrs; avoid screens 30 minutes before sleep.","#f59e0b"))
        if notifications > 150:
            recs.append(("🔔","Reduce Notification Load",
                f"{notifications} notifications/day creates constant distraction. Batch-check messages 2–3× daily.","#f59e0b"))
        if weekend_screen > 10:
            recs.append(("📅","Plan Offline Weekends",
                f"{weekend_screen:.1f} hrs on weekends is high. Schedule outdoor activities to cut device time.","#f59e0b"))
        if acad_enc == 0:
            recs.append(("📚","Protect Work / Study Focus",
                "Smartphone usage is impacting academic/work performance. Try Pomodoro with phone in another room.","#7c3aed"))
        if gaming > 3:
            recs.append(("🎮","Manage Gaming Sessions",
                f"{gaming:.1f} hrs/day. Limit to 45-min blocks with mandatory 15-min breaks.","#f59e0b"))
        if not label:
            recs.append(("✅","Maintain Healthy Habits",
                "Great balance! Schedule regular digital detox days to keep your healthy relationship with technology.","#10b981"))
        if not recs:
            recs.append(("🌟","Looking Good",
                "Usage patterns look healthy. Continue monitoring with regular self-assessments.","#10b981"))

        for icon, title, body, c in recs[:6]:
            st.markdown(f"""
            <div style='background:#1a1a26;border:1px solid #2a2a3e;border-left:3px solid {c};
            border-radius:12px;padding:18px;margin-bottom:12px;display:flex;gap:14px;align-items:flex-start;'>
              <div style='font-size:1.5rem;flex-shrink:0;margin-top:2px;'>{icon}</div>
              <div>
                <div style='font-weight:700;font-size:0.92rem;color:#f0f0ff;margin-bottom:6px;'>{title}</div>
                <div style='font-size:0.8rem;color:#c4c4d8;line-height:1.6;'>{body}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════
    # TAB 4 — HISTORY
    # ════════════════════════════════════
    with tab4:
        if not st.session_state.history:
            st.markdown("<div style='text-align:center;padding:40px;color:#6b6b8a;font-family:Space Mono,monospace;font-size:0.8rem;'>No prediction history yet.</div>", unsafe_allow_html=True)
        else:
            hist_df = pd.DataFrame(st.session_state.history)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=list(range(1, len(hist_df)+1)), y=hist_df["prob"],
                mode="lines+markers",
                line=dict(color="#7c3aed", width=2.5),
                marker=dict(size=10, color=[risk_color(p) for p in hist_df["prob"]],
                            line=dict(color="#0a0a0f", width=2)),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
            ))
            fig_hist.add_hline(y=50, line_dash="dash", line_color="#6b6b8a",
                               annotation_text="Decision Threshold (50%)",
                               annotation_font=dict(family="Space Mono", size=10, color="#6b6b8a"))
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(family="Space Mono",size=10,color="#6b6b8a"),
                           gridcolor="#1a1a26",
                           title=dict(text="Run", font=dict(family="Space Mono",size=10,color="#6b6b8a"))),
                yaxis=dict(range=[0,105],
                           tickfont=dict(family="Space Mono",size=10,color="#6b6b8a"),
                           gridcolor="#1a1a26",
                           title=dict(text="Risk Score (%)", font=dict(family="Space Mono",size=10,color="#6b6b8a"))),
                margin=dict(t=10,b=40,l=50,r=20), height=280, showlegend=False,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            display_df = hist_df[["prob","label","source","screen","social","sleep"]].copy()
            display_df.columns = ["Risk %","Label","Source","Screen (hrs)","Social (hrs)","Sleep (hrs)"]
            display_df["Label"] = display_df["Label"].map({1:"Addicted",0:"Not Addicted"})
            display_df.index = [f"Run {i+1}" for i in range(len(display_df))]
            st.dataframe(
                display_df.style.background_gradient(subset=["Risk %"], cmap="RdYlGn_r"),
                use_container_width=True
            )
            if st.button("🗑️  Clear History"):
                st.session_state.history = []
                st.rerun()


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<hr style='margin:0 0 16px;'>
<div style='display:flex;justify-content:space-between;align-items:center;
font-family:Space Mono,monospace;font-size:0.65rem;color:#4a4a6a;'>
  <span>SmartSense · Stacking Ensemble · Dataset: 7,500 rows · 17 features</span>
  <span>Place smartphone_addiction_model.pkl alongside app.py to activate the model</span>
</div>
""", unsafe_allow_html=True)
