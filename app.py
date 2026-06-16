import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from predict import predict_neuropathy
from sensor import list_ports, simulate_recording, live_recording

st.set_page_config(
    page_title="Neuropathy Screening",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.01em;
  }

  /* Background with gradient orbs */
  .stApp {
    background:
      radial-gradient(ellipse 55% 45% at 8% 12%,  rgba(94, 58, 255, 0.22) 0%, transparent 55%),
      radial-gradient(ellipse 50% 55% at 92% 88%,  rgba(0, 100, 230, 0.20) 0%, transparent 55%),
      radial-gradient(ellipse 40% 35% at 70% 22%,  rgba(48, 209, 88, 0.08) 0%, transparent 50%),
      #07070d;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding-top: 60px;
    padding-bottom: 80px;
    max-width: 660px;
  }

  /* Glass cards — target st.container(border=True) */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 18px !important;
    box-shadow:
      0 4px 32px rgba(0, 0, 0, 0.35),
      inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    padding: 4px !important;
    margin-bottom: 12px !important;
  }

  /* Make inner vertical block transparent */
  [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] {
    background: transparent !important;
    padding: 20px 22px !important;
  }

  /* Section label */
  .section-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.35);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 16px;
  }

  /* Page title */
  .page-title {
    font-family: "Syne", sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f5f5f7;
    letter-spacing: -0.04em;
    line-height: 1.06;
    margin-bottom: 36px;
  }

  /* Inputs */
  .stNumberInput input {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.10) !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    backdrop-filter: blur(8px) !important;
  }
  .stNumberInput input:focus {
    border-color: rgba(0, 113, 227, 0.8) !important;
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.18) !important;
  }
  .stNumberInput button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #f5f5f7 !important;
    border-radius: 8px !important;
  }

  /* Labels */
  label[data-testid="stWidgetLabel"] p,
  .stRadio [data-testid="stWidgetLabel"] p {
    color: rgba(255, 255, 255, 0.65) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
  }

  /* Radio */
  .stRadio label span {
    color: #ebebf0 !important;
    font-size: 0.9rem !important;
  }
  .stRadio [data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.82rem !important;
  }

  /* Segmented control */
  [data-testid="stSegmentedControl"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 3px !important;
    backdrop-filter: blur(8px) !important;
  }
  [data-testid="stSegmentedControl"] button {
    color: rgba(255, 255, 255, 0.55) !important;
    font-size: 0.855rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
  }
  [data-testid="stSegmentedControl"] button[aria-selected="true"] {
    background: rgba(255, 255, 255, 0.12) !important;
    color: #f5f5f7 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
  }

  /* Select box */
  .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #f5f5f7 !important;
    backdrop-filter: blur(8px) !important;
  }

  /* Primary button */
  .stButton > button[kind="primary"] {
    background: rgba(0, 113, 227, 0.9) !important;
    backdrop-filter: blur(12px) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 130, 255, 0.5) !important;
    border-radius: 980px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 13px 28px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 16px rgba(0, 113, 227, 0.35) !important;
    transition: all 0.15s ease !important;
    height: auto !important;
    width: 100% !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: rgba(0, 125, 245, 0.95) !important;
    box-shadow: 0 4px 24px rgba(0, 113, 227, 0.5) !important;
  }
  .stButton > button[kind="primary"]:disabled {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.2) !important;
    box-shadow: none !important;
  }

  /* Secondary button */
  .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(8px) !important;
    color: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 980px !important;
    font-size: 0.855rem !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    box-shadow: none !important;
    transition: all 0.15s ease !important;
  }
  .stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.11) !important;
  }

  /* Progress bar */
  .stProgress > div > div {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 4px !important;
    height: 3px !important;
  }
  .stProgress > div > div > div {
    background: #0071e3 !important;
    border-radius: 4px !important;
  }

  /* Info / warning text */
  [data-testid="stAlert"] {
    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.855rem !important;
  }

  /* Caption */
  .stCaption p {
    color: rgba(255,255,255,0.28) !important;
    font-size: 0.78rem !important;
  }

  /* Divider */
  hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 24px 0 !important;
  }

  /* Result cards */
  .result-card {
    border-radius: 20px;
    padding: 40px 32px 32px;
    text-align: center;
    backdrop-filter: blur(32px);
    -webkit-backdrop-filter: blur(32px);
  }
  .result-healthy {
    background: rgba(48, 209, 88, 0.08);
    border: 1px solid rgba(48, 209, 88, 0.25);
    box-shadow: 0 0 60px rgba(48, 209, 88, 0.08), 0 4px 32px rgba(0,0,0,0.4);
  }
  .result-risk {
    background: rgba(255, 69, 58, 0.08);
    border: 1px solid rgba(255, 69, 58, 0.25);
    box-shadow: 0 0 60px rgba(255, 69, 58, 0.08), 0 4px 32px rgba(0,0,0,0.4);
  }
  .result-verdict-healthy {
    font-family: "Syne", sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #30d158;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 8px;
  }
  .result-verdict-risk {
    font-family: "Syne", sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ff453a;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 8px;
  }
  .result-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    margin-bottom: 28px;
  }
  .prob-row {
    display: flex;
    justify-content: center;
    gap: 0;
    margin: 0 auto 24px;
    max-width: 360px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow: hidden;
  }
  .prob-item {
    flex: 1;
    padding: 16px 12px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
  }
  .prob-item:last-child { border-right: none; }
  .prob-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: #f5f5f7;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 4px;
  }
  .prob-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }
  .input-row {
    display: flex;
    justify-content: center;
    gap: 0;
    max-width: 420px;
    margin: 0 auto;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    overflow: hidden;
  }
  .input-item {
    flex: 1;
    padding: 12px 8px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  .input-item:last-child { border-right: none; }
  .input-value {
    font-size: 0.88rem;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    line-height: 1;
    margin-bottom: 4px;
  }
  .input-key {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.25);
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }
  .disclaimer {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.2);
    margin-top: 20px;
    letter-spacing: 0.01em;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Neuropathy Screening</div>', unsafe_allow_html=True)

# ── Patient information ────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    with col2:
        sex_str = st.radio("Sex", ["Male", "Female"], horizontal=True)

sex = 0 if sex_str == "Male" else 1

# ── EMG input ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="section-label">EMG Input</div>', unsafe_allow_html=True)

    mode = st.segmented_control(
        "mode",
        options=["Manual", "Simulate", "Live Sensor"],
        default="Manual",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    emg_rms, emg_freq = None, None

    if mode == "Manual":
        c1, c2 = st.columns(2)
        with c1:
            emg_rms  = st.number_input("EMG RMS (uV)",          min_value=0.0, value=104.0, step=0.1, format="%.2f")
        with c2:
            emg_freq = st.number_input("Median Frequency (Hz)", min_value=0.0, value=75.0,  step=0.1, format="%.2f")

    elif mode == "Simulate":
        st.markdown('<p style="font-size:0.855rem;color:rgba(255,255,255,0.38);margin:0 0 14px;">Generates 60 seconds of synthetic EMG data — no sensor required.</p>', unsafe_allow_html=True)

        if st.button("Run Simulation", type="primary"):
            bar = st.progress(0, text="Starting simulation...")

            def sim_cb(elapsed, total):
                bar.progress(elapsed / total, text=f"Simulating  —  {total - elapsed}s remaining")

            with st.spinner(""):
                rms, freq = simulate_recording(duration=60, callback=sim_cb)

            bar.progress(1.0, text="Complete")
            st.session_state["sim_rms"]  = rms
            st.session_state["sim_freq"] = freq

        if "sim_rms" in st.session_state:
            emg_rms  = st.session_state["sim_rms"]
            emg_freq = st.session_state["sim_freq"]
            st.markdown(f'<p style="font-size:0.78rem;color:rgba(255,255,255,0.28);margin-top:10px;">RMS: {emg_rms:.2f} uV &nbsp;&nbsp;·&nbsp;&nbsp; Median Frequency: {emg_freq:.2f} Hz</p>', unsafe_allow_html=True)

    elif mode == "Live Sensor":
        ports = list_ports()
        if not ports:
            st.markdown('<p style="font-size:0.855rem;color:rgba(255,153,10,0.85);margin:0 0 12px;">No serial ports detected. Connect the EMG sensor via USB.</p>', unsafe_allow_html=True)
            if st.button("Refresh Ports", type="secondary"):
                st.rerun()
        else:
            col_p, col_r = st.columns([3, 1])
            with col_p:
                port = st.selectbox("Port", ports, label_visibility="collapsed")
            with col_r:
                st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
                if st.button("Refresh", type="secondary"):
                    st.rerun()

            st.markdown(f'<p style="font-size:0.855rem;color:rgba(255,255,255,0.35);margin:0 0 14px;">Patient holds sensor. 60 seconds of EMG data will be recorded on <span style="color:rgba(255,255,255,0.65)">{port}</span>.</p>', unsafe_allow_html=True)

            if st.button("Start Recording", type="primary"):
                bar = st.progress(0, text="Connecting to sensor...")

                def live_cb(elapsed, total):
                    bar.progress(elapsed / total, text=f"Recording  —  {total - elapsed}s remaining")

                try:
                    with st.spinner(""):
                        rms, freq = live_recording(port, duration=60, callback=live_cb)
                    bar.progress(1.0, text="Recording complete")
                    st.session_state["live_rms"]  = rms
                    st.session_state["live_freq"] = freq
                except Exception as e:
                    st.markdown(f'<p style="font-size:0.855rem;color:#ff453a;">Error: {e}</p>', unsafe_allow_html=True)

            if "live_rms" in st.session_state:
                emg_rms  = st.session_state["live_rms"]
                emg_freq = st.session_state["live_freq"]
                st.markdown(f'<p style="font-size:0.78rem;color:rgba(255,255,255,0.28);margin-top:10px;">RMS: {emg_rms:.2f} uV &nbsp;&nbsp;·&nbsp;&nbsp; Median Frequency: {emg_freq:.2f} Hz</p>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
predict_ready = emg_rms is not None and emg_freq is not None

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

if st.button("Predict", type="primary", use_container_width=True, disabled=not predict_ready):
    result = predict_neuropathy(
        age=float(age), sex=sex,
        emg_rms=float(emg_rms),
        emg_median_freq=float(emg_freq)
    )

    label    = result['prediction']
    conf     = result['confidence'] * 100
    p_health = result['prob_healthy'] * 100
    p_risk   = result['prob_risk']   * 100

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if label == "Healthy":
        verdict_class = "result-healthy"
        verdict_text_class = "result-verdict-healthy"
        verdict_text = "Healthy"
        sub_text = "No neuropathy risk detected"
        prob_healthy_style = 'style="color:#30d158"'
        prob_risk_style = ""
    else:
        verdict_class = "result-risk"
        verdict_text_class = "result-verdict-risk"
        verdict_text = "Neuropathy Risk"
        sub_text = "Elevated risk of peripheral neuropathy detected"
        prob_healthy_style = ""
        prob_risk_style = 'style="color:#ff453a"'

    st.markdown(f"""
    <div class="result-card {verdict_class}">
      <div class="{verdict_text_class}">{verdict_text}</div>
      <div class="result-sub">{sub_text}</div>
      <div class="prob-row">
        <div class="prob-item">
          <div class="prob-value" {prob_healthy_style}>{p_health:.1f}%</div>
          <div class="prob-label">Healthy</div>
        </div>
        <div class="prob-item">
          <div class="prob-value" {prob_risk_style}>{p_risk:.1f}%</div>
          <div class="prob-label">At Risk</div>
        </div>
        <div class="prob-item">
          <div class="prob-value">{conf:.1f}%</div>
          <div class="prob-label">Confidence</div>
        </div>
      </div>
      <div class="input-row">
        <div class="input-item">
          <div class="input-value">{int(age)}</div>
          <div class="input-key">Age</div>
        </div>
        <div class="input-item">
          <div class="input-value">{sex_str}</div>
          <div class="input-key">Sex</div>
        </div>
        <div class="input-item">
          <div class="input-value">{emg_rms:.1f} uV</div>
          <div class="input-key">EMG RMS</div>
        </div>
        <div class="input-item">
          <div class="input-value">{emg_freq:.1f} Hz</div>
          <div class="input-key">Median Freq</div>
        </div>
      </div>
      <div class="disclaimer">For screening purposes only. Does not replace clinical diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)
