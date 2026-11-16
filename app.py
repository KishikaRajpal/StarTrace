"""
app.py - StarTrace: AI Exoplanet Discovery & Transit Classification Web Dashboard
Interactive, live-demo-ready Streamlit interface powered by PyTorch Dual-Branch 1D-CNN
and NASA TESS light curve streaming.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import (
    process_lightcurve,
    PRESET_METADATA,
    generate_synthetic_lightcurve
)
from model import (
    CLASSES,
    CLASS_DESCRIPTIONS,
    CLASS_COLORS,
    get_model,
    classify_lightcurve
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="StarTrace | AI Exoplanet Discovery",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Deep-Space Dark CSS Styling
# ---------------------------------------------------------
DARK_THEME_CSS = """
<style>
/* Main Container & Background */
.stApp {
    background-color: #090d16;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15);
}

/* Header Text Gradient */
.brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #9055ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 0px;
}

.brand-subtitle {
    font-size: 0.85rem;
    color: #8da2c0;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(16, 24, 42, 0.75);
    border: 1px solid rgba(0, 242, 254, 0.15);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(0, 242, 254, 0.4);
}

/* KPI Card */
.kpi-title {
    font-size: 0.78rem;
    color: #7e9bb6;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.2;
}

.kpi-unit {
    font-size: 0.85rem;
    font-weight: 400;
    color: #00f2fe;
    margin-left: 0.25rem;
}

/* Alert Banner */
.alert-card {
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}

.alert-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #00e676);
}

.alert-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.verdict-tag {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

.conf-pill {
    font-size: 0.95rem;
    font-weight: 700;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.1);
}

/* Streamlit Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00f2fe 0%, #0072ff 100%) !important;
    color: #060911 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.35) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6) !important;
    transform: translateY(-1px) !important;
}

/* Streamlit Widget Text */
div[data-testid="stMarkdownContainer"] p {
    color: #cbd5e1;
}
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Pipeline Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing PyTorch Dual-Branch CNN...")
def load_cached_model():
    return get_model("weights.pth")


@st.cache_data(show_spinner=False)
def run_pipeline_cached(source_type, tic_id, window_length, min_period, max_period):
    return process_lightcurve(
        source_type=source_type,
        tic_id=tic_id,
        csv_file=None,
        window_length=window_length,
        min_period=min_period,
        max_period=max_period
    )


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-title">🪐 StarTrace</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">AI Exoplanet Discovery Engine</div>', unsafe_allow_html=True)

    st.markdown("### 📡 Ingestion Source")
    input_mode = st.radio(
        "Select Telemetry Source",
        ["NASA MAST Live Stream", "Upload CSV"],
        label_visibility="collapsed"
    )

    preset_name_map = {
        "Kepler-1649c (TIC 25155310) — Confirmed Exoplanet": "25155310",
        "Eclipsing Binary (TIC 149603524) — Binary System": "149603524",
        "Stellar Blend (TIC 38846515) — Background Blend": "38846515",
        "Variable Star (TIC 261136638) — Stellar Noise": "261136638"
    }

    selected_tic = "25155310"
    uploaded_file = None

    if input_mode == "NASA MAST Live Stream":
        st.markdown("### 🎯 One-Click Demo Presets")
        preset_choice = st.selectbox(
            "Target Candidate Presets",
            list(preset_name_map.keys()),
            index=0
        )
        default_tic = preset_name_map[preset_choice]

        st.markdown("### 🔭 Custom Star Query")
        selected_tic = st.text_input(
            "TESS Input Catalog (TIC) ID",
            value=default_tic,
            help="Enter any TIC numeric ID (e.g. 25155310) to stream directly from NASA MAST"
        )
    else:
        st.markdown("### 📂 Upload Light Curve")
        uploaded_file = st.file_uploader(
            "Upload Photometry CSV",
            type=["csv", "txt"],
            help="CSV containing 'time' and 'flux' columns"
        )

    st.markdown("### ⚙️ Signal Conditioning")
    sg_window = st.slider(
        "Savitzky-Golay Filter Window",
        min_value=51,
        max_value=201,
        value=101,
        step=2,
        help="Cadence window for detrending low-frequency stellar activity"
    )

    with st.expander("🛠️ Advanced Periodicity Bounds", expanded=False):
        min_p = st.number_input("Min Period (Days)", min_value=0.2, max_value=5.0, value=0.5, step=0.1)
        max_p = st.number_input("Max Period (Days)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

    run_pipeline = st.button("⚡ Run Analysis Pipeline")

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.78rem; color: #64748b; line-height: 1.4;'>"
        "<b>Model:</b> Dual-Branch 1D-CNN (2001 Global + 201 Local)<br>"
        "<b>Mission:</b> NASA TESS 2-min Cadence (SPOC)<br>"
        "<b>Search:</b> Astropy Box Least Squares (BLS)"
        "</div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Pipeline Execution & Processing
# ---------------------------------------------------------
model = load_cached_model()

# Auto-run Kepler-1649c on first visit or when user clicks run
should_process = run_pipeline or ("pipeline_result" not in st.session_state)

if should_process:
    with st.spinner("Streaming & conditioning TESS telemetry..."):
        if input_mode == "Upload CSV" and uploaded_file is not None:
            result = process_lightcurve(
                source_type="Upload CSV",
                csv_file=uploaded_file,
                window_length=sg_window,
                min_period=min_p,
                max_period=max_p
            )
        else:
            result = run_pipeline_cached(
                source_type="NASA MAST Live Stream",
                tic_id=selected_tic,
                window_length=sg_window,
                min_period=min_p,
                max_period=max_p
            )

        # Run Neural Classification
        classification = classify_lightcurve(
            model,
            result["dual_views"]["global_view"],
            result["dual_views"]["local_view"],
            bls_stats=result["bls_stats"]
        )

        st.session_state["pipeline_result"] = result
        st.session_state["classification"] = classification

res = st.session_state["pipeline_result"]
clf = st.session_state["classification"]
bls = res["bls_stats"]

# ---------------------------------------------------------
# Top Status Bar & Metadata
# ---------------------------------------------------------
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    badge_color = "#00e676" if not res["is_fallback"] else "#ffd166"
    badge_label = "LIVE NASA MAST STREAM" if not res["is_fallback"] else "OFFLINE SYNTHETIC CACHE"
    st.markdown(
        f"<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;'>"
        f"<span style='background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}66; "
        f"font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.8px;'>"
        f"{badge_label}</span>"
        f"<span style='color: #94a3b8; font-size: 0.85rem;'>Target: <b>{res['source_label']}</b></span>"
        f"</div>",
        unsafe_allow_html=True
    )

with top_col2:
    st.markdown(
        f"<div style='text-align: right; color: #64748b; font-size: 0.8rem;'>"
        f"Cadence Points: <b>{len(res['time']):,}</b></div>",
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Classification Alert Banner
# ---------------------------------------------------------
pred_class = clf["prediction"]
top_conf = clf["confidence"] * 100.0
card_color = CLASS_COLORS.get(pred_class, "#00f2fe")
card_desc = CLASS_DESCRIPTIONS.get(pred_class, "")

icon_map = {
    "Exoplanet Transit": "🪐",
    "Eclipsing Binary": "✨",
    "Background Blend": "🔍",
    "Stellar Noise / Activity": "⚡"
}
icon = icon_map.get(pred_class, "🌌")

st.markdown(
    f"""
    <div class="glass-card alert-card" style="border-left: 4px solid {card_color}; background: rgba(16, 24, 42, 0.9);">
        <div class="alert-title-row">
            <div class="verdict-tag" style="color: {card_color};">
                {icon} {pred_class.upper()}
            </div>
            <div class="conf-pill" style="color: {card_color}; border: 1px solid {card_color}55;">
                CONFIDENCE: {top_conf:.1f}%
            </div>
        </div>
        <div style="color: #cbd5e1; font-size: 0.92rem; margin-bottom: 1rem; line-height: 1.5;">
            {card_desc}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Horizontal Multi-Class Softmax Probability Breakdown
# ---------------------------------------------------------
st.markdown("<div style='font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: #8da2c0; margin-bottom: 0.6rem; letter-spacing: 0.5px;'>Neural Network Confidence Distribution</div>", unsafe_allow_html=True)

prob_cols = st.columns(4)
for i, c_name in enumerate(CLASSES):
    p_val = clf["probabilities"][c_name] * 100.0
    c_color = CLASS_COLORS[c_name]
    with prob_cols[i]:
        st.markdown(
            f"""
            <div style="background: rgba(13, 19, 34, 0.7); border: 1px solid {c_color}33; border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                    <span style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">{c_name}</span>
                    <span style="font-size: 0.85rem; color: {c_color}; font-weight: 700;">{p_val:.1f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; width: 100%; overflow: hidden;">
                    <div style="background: {c_color}; height: 100%; width: {max(2.0, p_val)}%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# 4 KPI Metric Cards
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-title">Orbital Period (P)</div>
            <div class="kpi-value">{bls['period_days']:.4f}<span class="kpi-unit">days</span></div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">BLS Peak Harmonics</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-title">Transit Depth (δ)</div>
            <div class="kpi-value">{bls['depth_pct']:.3f}<span class="kpi-unit">%</span></div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">{(bls['depth_pct'] * 10000):.0f} ppm drop</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-title">Transit Duration (T)</div>
            <div class="kpi-value">{bls['duration_hours']:.2f}<span class="kpi-unit">hrs</span></div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">Ingress to Egress</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    snr_color = "#00e676" if bls['snr'] >= 10.0 else ("#ffd166" if bls['snr'] >= 7.1 else "#ff5252")
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="kpi-title">Detection SNR</div>
            <div class="kpi-value" style="color: {snr_color};">{bls['snr']:.1f}<span class="kpi-unit">σ</span></div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.3rem;">Threshold: 7.1σ (NASA TESS)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Chart 1: Raw vs Detrended Flux Time-Series
# ---------------------------------------------------------
st.markdown("### 📈 Flux Telemetry & Savitzky-Golay Detrending")

fig_ts = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Raw Ingested Flux with Stellar Activity Trend", "Detrended & Normalized Baseline Flux (1.0)")
)

# Downsample for lightning-fast rendering if points > 8000
step = max(1, len(res["time"]) // 6000)
t_plot = res["time"][::step]
raw_plot = res["raw_flux"][::step]
trend_plot = res["trend"][::step]
detr_plot = res["detrended_flux"][::step]

fig_ts.add_trace(
    go.Scatter(
        x=t_plot, y=raw_plot,
        mode="markers",
        marker=dict(size=2.5, color="#64748b", opacity=0.45),
        name="Raw Photometry",
        hoverinfo="x+y"
    ),
    row=1, col=1
)

fig_ts.add_trace(
    go.Scatter(
        x=t_plot, y=trend_plot,
        mode="lines",
        line=dict(color="#ffd166", width=2.0),
        name="Stellar Activity Trend",
        hoverinfo="skip"
    ),
    row=1, col=1
)

fig_ts.add_trace(
    go.Scatter(
        x=t_plot, y=detr_plot,
        mode="markers",
        marker=dict(size=2.5, color="#00f2fe", opacity=0.6),
        name="Detrended Flux",
        hoverinfo="x+y"
    ),
    row=2, col=1
)

fig_ts.add_hline(y=1.0, line_dash="dash", line_color="#4facfe", line_width=1.0, row=2, col=1)

fig_ts.update_layout(
    template="plotly_dark",
    plot_bgcolor="#090d16",
    paper_bgcolor="#090d16",
    height=480,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest"
)
fig_ts.update_xaxes(title_text="BJD - 2457000 (Time in Days)", row=2, col=1, gridcolor="#1e293b")
fig_ts.update_xaxes(gridcolor="#1e293b", row=1, col=1)
fig_ts.update_yaxes(title_text="Relative Flux", row=1, col=1, gridcolor="#1e293b")
fig_ts.update_yaxes(title_text="Normalized Flux", row=2, col=1, gridcolor="#1e293b")

st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------
# Plotly Charts 2 & 3: BLS Periodogram & Phase-Folded Views
# ---------------------------------------------------------
col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("### 📊 Box Least Squares (BLS) Periodogram")

    p_periods = res["periodogram"]["periods"]
    p_powers = res["periodogram"]["powers"]

    fig_bls = go.Figure()
    fig_bls.add_trace(
        go.Scatter(
            x=p_periods, y=p_powers,
            mode="lines",
            line=dict(color="#4facfe", width=1.8),
            name="BLS Power Spectrum"
        )
    )

    # Vertical line marking peak period
    fig_bls.add_vline(
        x=bls["period_days"],
        line_width=2,
        line_dash="dot",
        line_color="#00f2fe",
        annotation_text=f"P = {bls['period_days']:.3f} d",
        annotation_position="top right",
        annotation_font=dict(color="#00f2fe", size=11)
    )

    fig_bls.update_layout(
        template="plotly_dark",
        plot_bgcolor="#090d16",
        paper_bgcolor="#090d16",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Orbital Trial Period (Days)", gridcolor="#1e293b"),
        yaxis=dict(title="BLS Power / SNR", gridcolor="#1e293b"),
        showlegend=False
    )
    st.plotly_chart(fig_bls, use_container_width=True)

with col_p2:
    view_tab1, view_tab2 = st.tabs(["🔍 Local Transit Zoom (201 Bins)", "🌐 Full Orbit Global (2001 Bins)"])

    raw_phase = res["dual_views"]["phase_raw"]
    raw_flux_folded = res["dual_views"]["flux_raw"]

    with view_tab1:
        # Local view [-0.05, 0.05]
        local_mask = (raw_phase >= -0.05) & (raw_phase <= 0.05)
        phase_local_raw = raw_phase[local_mask]
        flux_local_raw = raw_flux_folded[local_mask]

        fig_local = go.Figure()

        # Downsample scatter points for speed
        l_step = max(1, len(phase_local_raw) // 1500)
        fig_local.add_trace(
            go.Scatter(
                x=phase_local_raw[::l_step],
                y=flux_local_raw[::l_step],
                mode="markers",
                marker=dict(size=3.0, color="#64748b", opacity=0.45),
                name="Folded Photometry"
            )
        )

        # 201-point binned median line
        fig_local.add_trace(
            go.Scatter(
                x=res["dual_views"]["centers_local"],
                y=res["dual_views"]["local_view"],
                mode="lines+markers",
                line=dict(color="#00f2fe", width=2.5),
                marker=dict(size=3.5, color="#00f2fe"),
                name="Median Binned (201)"
            )
        )

        fig_local.update_layout(
            template="plotly_dark",
            plot_bgcolor="#090d16",
            paper_bgcolor="#090d16",
            height=340,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(title="Orbital Phase (Centered on t₀)", range=[-0.05, 0.05], gridcolor="#1e293b"),
            yaxis=dict(title="Normalized Flux", gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with view_tab2:
        fig_global = go.Figure()
        g_step = max(1, len(raw_phase) // 2500)
        fig_global.add_trace(
            go.Scatter(
                x=raw_phase[::g_step],
                y=raw_flux_folded[::g_step],
                mode="markers",
                marker=dict(size=2.0, color="#475569", opacity=0.35),
                name="Folded Photometry"
            )
        )

        fig_global.add_trace(
            go.Scatter(
                x=res["dual_views"]["centers_global"],
                y=res["dual_views"]["global_view"],
                mode="lines",
                line=dict(color="#ff007f", width=1.8),
                name="Median Binned (2001)"
            )
        )

        fig_global.update_layout(
            template="plotly_dark",
            plot_bgcolor="#090d16",
            paper_bgcolor="#090d16",
            height=340,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(title="Full Orbital Phase [-0.5, 0.5]", range=[-0.5, 0.5], gridcolor="#1e293b"),
            yaxis=dict(title="Normalized Flux", gridcolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_global, use_container_width=True)

# ---------------------------------------------------------
# Footer & Scientific Citation
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 0.5rem 0;'>"
    "StarTrace AI Exoplanet Discovery Platform &bull; Powered by NASA TESS Telemetry &bull; PyTorch 1D-CNN Architecture"
    "</div>",
    unsafe_allow_html=True
)
