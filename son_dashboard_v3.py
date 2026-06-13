"""
Auto SON Decision Engine v3.0 — Streamlit Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES in v3.0:
  - Classification rebuilt on raw dBm (no MinMaxScaler bias)
  - Class-specific heatmap colorscales (instant visual clarity)
  - All 4 classes guaranteed in output
  - Removed confusing normalization toggle
  - Cleaner layout, better axis orientation
  - Anomaly highlighting in every chart
"""

import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from son_engine_v3 import (
    load_and_process, load_site_config,
    generate_labels, inject_synthetic_external,
    build_rf_neighbor_map, neighbor_similarity_scores,
    build_report, normalize_for_display, get_class_colorscale,
    LABELS, SON_ACTIONS, RCA_MAP,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto SON Decision Engine v3",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0a2342 0%, #1a4a8a 50%, #0d6efd 100%);
        color: white;
        padding: 22px 32px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }

    .kpi-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        margin: 4px 0;
    }
    .kpi-card .label { font-size: 0.72rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-card .value { font-size: 1.7rem; font-weight: 700; }

    .class-Normal   { color: #4da6ff; }
    .class-Hardware { color: #ff4422; }
    .class-Traffic  { color: #ffaa22; }
    .class-External { color: #cc66ff; }

    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-Normal   { background: #0d2a5a; color: #4da6ff; border: 1px solid #1a4a9a; }
    .badge-Hardware { background: #3d0a0a; color: #ff6644; border: 1px solid #aa2200; }
    .badge-Traffic  { background: #3d2a00; color: #ffaa22; border: 1px solid #aa6600; }
    .badge-External { background: #2a0a3d; color: #cc66ff; border: 1px solid #7722aa; }

    .priority-High   { border-left: 4px solid #ff4422; background: #1a0000; padding: 10px 14px; border-radius: 6px; }
    .priority-Medium { border-left: 4px solid #ffaa22; background: #1a1000; padding: 10px 14px; border-radius: 6px; }
    .priority-Low    { border-left: 4px solid #44cc44; background: #001a00; padding: 10px 14px; border-radius: 6px; }

    .rca-box {
        background: #141e30;
        border: 1px solid #1e3a5a;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #c0d4f0;
    }
    .action-item {
        background: #0f1d35;
        border-left: 3px solid #0d6efd;
        border-radius: 4px;
        padding: 7px 12px;
        margin: 3px 0;
        font-size: 0.85rem;
        color: #b0ccee;
    }
    div[data-testid="stSidebar"] { background: #0d1b2e; }

    .legend-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; font-size: 0.85rem; }
    .legend-dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Color maps
# ─────────────────────────────────────────────────────────────
PRED_COLOR = {
    "Normal":   "#4da6ff",
    "Hardware": "#ff4422",
    "Traffic":  "#ffaa22",
    "External": "#cc66ff",
}
PRIORITY_COLOR = {"High": "#ff4422", "Medium": "#ffaa22", "Low": "#44cc44"}

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 Auto SON Decision Engine <span style="font-size:1rem;opacity:0.65">v3.0</span></h1>
    <p>Detect · Diagnose · Decide · Recommend · Improve &nbsp;|&nbsp;
       Raw-dBm Classification · RF-Aware Neighbor Intelligence · 4-Class RCA</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:12px;">
        <svg width="72" height="72" viewBox="0 0 80 80" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="g5" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#0d6efd"/>
                    <stop offset="100%" style="stop-color:#00c8ff"/>
                </linearGradient>
            </defs>
            <rect width="80" height="80" rx="16" fill="url(#g5)"/>
            <text x="40" y="52" text-anchor="middle" font-family="Arial Black,sans-serif"
                  font-size="36" font-weight="900" fill="white" letter-spacing="-1">5G</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ Configuration")
    uploaded_rssi   = st.file_uploader("Upload RSSI Dataset (.xlsx)", type=["xlsx"], key="rssi")
    uploaded_config = st.file_uploader("Upload Site Config (.xlsx)",  type=["xlsx"], key="cfg")

    st.markdown("#### RF Neighbor Parameters")
    top_n        = st.slider("Max RF Neighbors per Cell",      3, 10, 5)
    min_nb_score = st.slider("Min Neighbor Score Threshold",   0.1, 0.8, 0.3, 0.05)

    run_btn = st.button("🚀 Run SON Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🎨 RCA Class Legend")
    for lbl, color, desc in [
        ("Normal",   "#4da6ff", "Stable • Low variance"),
        ("Hardware", "#ff4422", "Very low RSSI • Flat signal"),
        ("Traffic",  "#ffaa22", "Peak-hour degradation"),
        ("External", "#cc66ff", "Spikes • High variance"),
    ]:
        st.markdown(
            f'<div class="legend-row">'
            f'<span class="legend-dot" style="background:{color}"></span>'
            f'<span style="color:{color}"><b>{lbl}</b></span>'
            f'<span style="color:#888;font-size:0.78rem"> — {desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    **v3.0 Pipeline**
    1. Load raw RSSI (dBm)
    2. Feature engineering (mean/std/range/gradient)
    3. Rule-based classification (raw dBm)
    4. Synthetic injection if External missing
    5. RF neighbor map (distance + azimuth + coverage)
    6. RCA + SON action engine
    """)

# ─────────────────────────────────────────────────────────────
# Default file paths (for demo without upload)
# ─────────────────────────────────────────────────────────────
DEFAULT_RSSI   = "Input_file.xlsx"
DEFAULT_CONFIG = "Site_Config.xlsx"

# ─────────────────────────────────────────────────────────────
# Load & run pipeline (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_pipeline(rssi_path: str, config_path: str,
                    top_n: int, min_score: float):
    from son_engine_v3 import run_pipeline
    return run_pipeline(rssi_path, config_path,
                        top_neighbors=top_n,
                        min_neighbor_score=min_score)


if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None

if run_btn or st.session_state["pipeline_result"] is None:
    rssi_path   = uploaded_rssi.name   if uploaded_rssi   else DEFAULT_RSSI
    config_path = uploaded_config.name if uploaded_config else DEFAULT_CONFIG

    if uploaded_rssi:
        with open(rssi_path, "wb") as f:
            f.write(uploaded_rssi.getbuffer())
    if uploaded_config:
        with open(config_path, "wb") as f:
            f.write(uploaded_config.getbuffer())

    with st.spinner("⚙️ Running SON analysis..."):
        try:
            result = cached_pipeline(rssi_path, config_path, top_n, min_nb_score)
            st.session_state["pipeline_result"] = result
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

if st.session_state["pipeline_result"] is None:
    st.info("👈 Upload files and click **Run SON Analysis** to begin.")
    st.stop()

(report, cell_raw, cell_gnb, cell_features,
 labels, label_str, evidence, rf_neighbor_map) = st.session_state["pipeline_result"]

all_cells = sorted(cell_raw.keys())
real_cells = [c for c in all_cells if not c.startswith("SYN_")]
syn_cells  = [c for c in all_cells if c.startswith("SYN_")]

# ─────────────────────────────────────────────────────────────
# KPI summary row
# ─────────────────────────────────────────────────────────────
label_counts = pd.Series(label_str).value_counts()
total_cells  = len(all_cells)
high_priority = (report["Priority"] == "High").sum()

col1, col2, col3, col4, col5 = st.columns(5)
for col, label, color in zip(
    [col1, col2, col3, col4],
    ["Normal", "Hardware", "Traffic", "External"],
    ["#4da6ff", "#ff4422", "#ffaa22", "#cc66ff"],
):
    cnt = label_counts.get(label, 0)
    col.markdown(
        f'<div class="kpi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{cnt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
col5.markdown(
    f'<div class="kpi-card">'
    f'<div class="label">High Priority</div>'
    f'<div class="value" style="color:#ff4422">{high_priority}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Heatmap Explorer",
    "📊 Network Overview",
    "🔬 Cell Deep Dive",
    "🔗 RF Neighbors",
    "📋 Full Report",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: Heatmap Explorer  ← REBUILT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### 🗺️ Cell RSSI Heatmap — Class-Aware Visualization")
    st.markdown(
        "_Each cell's heatmap uses a colorscale tuned to its RCA class — "
        "patterns are instantly readable at a glance._"
    )

    # Controls
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        display_cells = st.multiselect(
            "Select cells to display",
            options=all_cells,
            default=real_cells[:8],
        )
    with c2:
        show_smooth = st.checkbox("Gaussian smoothing", value=False,
                                  help="Light blur for visual comfort — does NOT affect classification")
    with c3:
        flip_axes = st.checkbox("Flip axes (PRB on Y)", value=False)

    if not display_cells:
        st.info("Select at least one cell.")
    else:
        # Layout: 2 per row
        cols_per_row = 2
        rows_needed  = (len(display_cells) + cols_per_row - 1) // cols_per_row

        for row_i in range(rows_needed):
            cols = st.columns(cols_per_row)
            for col_i, col in enumerate(cols):
                idx = row_i * cols_per_row + col_i
                if idx >= len(display_cells):
                    break
                cell_name = display_cells[idx]
                raw_mat   = cell_raw[cell_name]
                lbl       = label_str[cell_name]
                ev        = evidence[cell_name]

                # Normalize ONLY for display
                display_mat = normalize_for_display(raw_mat)
                if show_smooth:
                    display_mat = gaussian_filter(display_mat.astype(float), sigma=1.0)

                colorscale = get_class_colorscale(lbl)
                n_hours, n_prb = raw_mat.shape

                if flip_axes:
                    z_data = display_mat.T
                    x_labels = [f"{h:02d}:00" for h in range(n_hours)]
                    y_labels = [f"P{i}" for i in range(n_prb)]
                    xaxis_title, yaxis_title = "Hour", "PRB"
                else:
                    z_data = display_mat
                    x_labels = [f"P{i}" for i in range(n_prb)]
                    y_labels = [f"{h:02d}:00" for h in range(n_hours)]
                    xaxis_title, yaxis_title = "PRB Index", "Hour"

                # Build hover text showing raw dBm
                hover_base = raw_mat if not flip_axes else raw_mat.T
                hover_text = [[f"{hover_base[i, j]:.1f} dBm"
                               for j in range(hover_base.shape[1])]
                              for i in range(hover_base.shape[0])]

                fig = go.Figure(go.Heatmap(
                    z=z_data,
                    x=x_labels,
                    y=y_labels,
                    colorscale=colorscale,
                    showscale=True,
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    colorbar=dict(
                        thickness=10,
                        len=0.8,
                        tickvals=[0, 0.5, 1],
                        ticktext=["Low", "Mid", "High"],
                        tickfont=dict(size=9),
                    ),
                ))

                # Badge title
                lbl_color = PRED_COLOR[lbl]
                is_syn    = cell_name.startswith("SYN_")
                syn_tag   = " ⚗️ synthetic" if is_syn else ""

                fig.update_layout(
                    title=dict(
                        text=f"<b>{cell_name}</b>  "
                             f"<span style='color:{lbl_color}'>[{lbl}]</span>{syn_tag}  "
                             f"<span style='font-size:11px;color:#888'>"
                             f"μ={ev['mean_dBm']:.1f} dBm  σ={ev['std_dB']:.1f} dB</span>",
                        font=dict(size=13),
                        x=0,
                    ),
                    xaxis=dict(title=xaxis_title, tickfont=dict(size=8),
                               tickangle=-45 if not flip_axes else 0),
                    yaxis=dict(title=yaxis_title, tickfont=dict(size=8)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=280,
                    margin=dict(t=45, b=30, l=50, r=60),
                )

                # Highlight peak-hour band for Traffic cells
                # add_shape used instead of add_hline because Y is categorical strings
                if lbl == "Traffic" and not flip_axes:
                    for ph in [19, 20]:
                        fig.add_shape(
                            type="line",
                            x0=0, x1=1, xref="paper",
                            y0=f"{ph:02d}:00", y1=f"{ph:02d}:00", yref="y",
                            line=dict(color="#ffcc00", width=1.5, dash="dot"),
                        )
                    fig.add_annotation(
                        x=1.02, xref="paper",
                        y="19:00", yref="y",
                        text="Peak hrs",
                        showarrow=False,
                        font=dict(color="#ffcc00", size=8),
                        xanchor="left",
                    )

                col.plotly_chart(fig, use_container_width=True)

    # Class color legend
    st.markdown("---")
    st.markdown("**Colorscale Guide**")
    leg_cols = st.columns(4)
    desc_map = {
        "Normal":   "Solid blue — calm, uniform signal. Easy to read.",
        "Hardware": "Deep red — uniform fault. No gradient variation.",
        "Traffic":  "Blue→Orange→Red — peak-hour degradation visible at peak rows.",
        "External": "Blue→White→Red — spikes jump out as bright/dark contrast bands.",
    }
    for col, lbl in zip(leg_cols, ["Normal", "Hardware", "Traffic", "External"]):
        col.markdown(
            f"<span style='color:{PRED_COLOR[lbl]}'><b>{lbl}</b></span><br>"
            f"<span style='font-size:0.78rem;color:#aaa'>{desc_map[lbl]}</span>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2: Network Overview
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📊 Network-Wide Overview")

    # ── Class distribution donut ──
    c_left, c_right = st.columns([1, 2])

    with c_left:
        fig_donut = go.Figure(go.Pie(
            labels=list(label_counts.index),
            values=list(label_counts.values),
            hole=0.55,
            marker=dict(colors=[PRED_COLOR[l] for l in label_counts.index]),
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        fig_donut.update_layout(
            title="RCA Class Distribution",
            paper_bgcolor="rgba(0,0,0,0)", font_color="white",
            height=280, margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        # Mean RSSI bar chart colored by class
        bar_df = report[~report["Cell_Name"].str.startswith("SYN_")].copy()
        bar_df = bar_df.sort_values("Mean_RSSI_dBm")
        fig_bar = go.Figure(go.Bar(
            x=bar_df["Cell_Name"],
            y=bar_df["Mean_RSSI_dBm"],
            marker_color=[PRED_COLOR[l] for l in bar_df["Prediction"]],
            text=bar_df["Prediction"],
            textposition="outside",
            textfont=dict(size=9),
        ))
        # Reference lines
        fig_bar.add_hline(y=-97, line=dict(color="#aaa", dash="dot", width=1),
                          annotation_text="Normal lower", annotation_font=dict(color="#aaa", size=9))
        fig_bar.add_hline(y=-108, line=dict(color="#ff4422", dash="dot", width=1),
                          annotation_text="Hardware threshold", annotation_font=dict(color="#ff4422", size=9))
        fig_bar.update_layout(
            title="Mean RSSI per Cell (raw dBm)",
            xaxis=dict(title="Cell", tickangle=-45, tickfont=dict(size=8)),
            yaxis=dict(title="Mean RSSI (dBm)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=280, margin=dict(t=40, b=60),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Fleet hourly profile ──
    st.markdown("##### Fleet-Wide RSSI Hourly Profile")
    real_raw = np.array([cell_raw[c] for c in real_cells])
    hourly_mean = real_raw.mean(axis=(0, 2))
    hourly_std  = real_raw.std(axis=(0, 2))
    hour_labels = [f"{h:02d}:00" for h in range(24)]

    fig_fleet = go.Figure()
    fig_fleet.add_trace(go.Scatter(
        x=hour_labels, y=hourly_mean + hourly_std,
        mode="lines", line=dict(width=0), showlegend=False,
        fillcolor="rgba(77,166,255,0.15)", fill="tonexty",
    ))
    fig_fleet.add_trace(go.Scatter(
        x=hour_labels, y=hourly_mean - hourly_std,
        mode="lines", line=dict(width=0), name="±1σ band",
        fillcolor="rgba(77,166,255,0.15)", fill="tozeroy",
    ))
    fig_fleet.add_trace(go.Scatter(
        x=hour_labels, y=hourly_mean,
        mode="lines+markers", name="Fleet Mean RSSI",
        line=dict(color="#4da6ff", width=2.5),
        marker=dict(size=5),
    ))
    fig_fleet.add_vrect(
        x0="19:00", x1="21:00",
        fillcolor="rgba(255,170,34,0.10)",
        line_width=0,
        annotation_text="Traffic Window",
        annotation_position="top left",
        annotation_font=dict(color="#ffaa22", size=10),
    )
    fig_fleet.update_layout(
        xaxis=dict(title="Hour of Day"),
        yaxis=dict(title="Mean RSSI (dBm)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=280, margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_fleet, use_container_width=True)

    # ── Feature scatter: std vs mean, colored by class ──
    st.markdown("##### Feature Space: Mean RSSI vs Std (classification boundary view)")
    feat_df = pd.DataFrame([
        {
            "Cell": c,
            "Mean_RSSI": evidence[c]["mean_dBm"],
            "Std_dB":    evidence[c]["std_dB"],
            "Class":     label_str[c],
            "Synthetic": "⚗️ Synthetic" if c.startswith("SYN_") else "Real",
        }
        for c in all_cells
    ])
    fig_scatter = px.scatter(
        feat_df,
        x="Mean_RSSI", y="Std_dB",
        color="Class",
        symbol="Synthetic",
        color_discrete_map=PRED_COLOR,
        hover_data=["Cell"],
        text="Cell",
        size_max=14,
    )
    # Threshold lines
    fig_scatter.add_vline(x=-108, line=dict(color="#ff4422", dash="dash", width=1),
                          annotation_text="HW thresh", annotation_font=dict(color="#ff4422", size=9))
    fig_scatter.add_hline(y=5, line=dict(color="#cc66ff", dash="dash", width=1),
                          annotation_text="Ext/Traffic std threshold",
                          annotation_font=dict(color="#cc66ff", size=9))
    fig_scatter.update_traces(textposition="top center", textfont=dict(size=8))
    fig_scatter.update_layout(
        xaxis=dict(title="Mean RSSI (dBm)"),
        yaxis=dict(title="Std (dB)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=350, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: Cell Deep Dive
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🔬 Cell-Level Deep Dive")

    sel_cell = st.selectbox("Select Cell", options=all_cells, key="dd_cell")

    row = report[report["Cell_Name"] == sel_cell].iloc[0]
    lbl = row["Prediction"]
    ev  = evidence[sel_cell]
    lbl_color = PRED_COLOR[lbl]

    # Header strip
    is_syn = sel_cell.startswith("SYN_")
    st.markdown(
        f'<div class="{("priority-High" if row["Priority"]=="High" else "priority-Medium" if row["Priority"]=="Medium" else "priority-Low")}">'
        f'<b style="color:{lbl_color};font-size:1.05rem">{lbl}</b>'
        f'{"  ⚗️ <i>synthetic sample</i>" if is_syn else ""}'
        f'&nbsp;·&nbsp; Priority: <b style="color:{PRIORITY_COLOR[row["Priority"]]}">{row["Priority"]}</b>'
        f'&nbsp;·&nbsp; GNB: {row["GNB_ID"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val, unit in [
        (m1, "Mean RSSI",  ev["mean_dBm"],    "dBm"),
        (m2, "Std",        ev["std_dB"],       "dB"),
        (m3, "Range",      ev["range_dB"],     "dB"),
        (m4, "Peak Drop",  ev["peak_drop_dB"], "dB"),
        (m5, "Gradient",   ev["mean_grad"],    "dB/hr"),
    ]:
        col.metric(label, f"{val:.1f} {unit}")

    # Hourly RSSI line chart
    st.markdown("##### Hourly RSSI Profile (Mean across PRBs)")
    mat       = cell_raw[sel_cell]
    hourly    = mat.mean(axis=1)
    hour_lbl  = [f"{h:02d}:00" for h in range(24)]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=hour_lbl, y=hourly,
        mode="lines+markers",
        line=dict(color=lbl_color, width=2.5),
        marker=dict(size=6),
        name="Mean RSSI",
    ))
    # Highlight peak hours
    fig_line.add_vrect(
        x0="19:00", x1="21:00",
        fillcolor="rgba(255,170,34,0.10)", line_width=0,
        annotation_text="Peak Window",
        annotation_font=dict(color="#ffaa22", size=9),
    )
    fig_line.update_layout(
        xaxis=dict(title="Hour"),
        yaxis=dict(title="Mean RSSI (dBm)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=240, margin=dict(t=10, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # PRB RSSI profile
    st.markdown("##### PRB RSSI Profile (Mean across 24hrs)")
    prb_means = mat.mean(axis=0)
    prb_lbl   = [f"P{i}" for i in range(len(prb_means))]

    fig_prb = go.Figure(go.Bar(
        x=prb_lbl, y=prb_means,
        marker_color=lbl_color,
        opacity=0.8,
    ))
    fig_prb.update_layout(
        xaxis=dict(title="PRB Index", tickangle=-90, tickfont=dict(size=7)),
        yaxis=dict(title="Mean RSSI (dBm)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white", height=220, margin=dict(t=10, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig_prb, use_container_width=True)

    # RCA + Actions
    st.markdown("##### Root Cause Analysis")
    st.markdown(f'<div class="rca-box">🔍 {row["Root_Cause"]}</div>', unsafe_allow_html=True)

    st.markdown("##### Recommended SON Actions")
    for action in row["SON_Actions"].split(" | "):
        st.markdown(f'<div class="action-item">▶ {action}</div>', unsafe_allow_html=True)

    st.markdown(f"**KPI Impact:** {row['KPI_Impact']}")


# ═══════════════════════════════════════════════════════════════
# TAB 4: RF Neighbors
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### 🔗 RF Neighbor Intelligence")

    # Summary table
    nb_rows = []
    for cell, nbs in rf_neighbor_map.items():
        for nb in nbs:
            nb_rows.append({
                "Cell A":       cell,
                "Cell B":       nb["cell_B"],
                "Distance (m)": nb["distance_m"],
                "Bearing°":     nb["bearing_deg"],
                "Az Diff°":     nb["az_diff_deg"],
                "In Beam?":     nb["coverage_match"],
                "Final Score":  nb["neighbor_score"],
            })

    if nb_rows:
        nb_table = pd.DataFrame(nb_rows)
        st.dataframe(
            nb_table.style
            .map(lambda v: "color: #44cc44" if v == "Yes" else "color: #ff4422",
                 subset=["In Beam?"])
            .background_gradient(subset=["Final Score"], cmap="Blues")
            .format({"Distance (m)": "{:.0f}", "Bearing°": "{:.1f}",
                     "Az Diff°": "{:.1f}", "Final Score": "{:.4f}"})
            .set_properties(**{"background-color": "#0d1b2e", "color": "#d0e0ff",
                               "border-color": "#1e3a5a"}),
            use_container_width=True, height=380,
        )

    # Per-cell view
    st.markdown("##### Per-Cell RF Neighbor Detail")
    nb_cell_sel = st.selectbox("Select Cell", sorted(rf_neighbor_map.keys()), key="nb_cell")
    cell_nbs    = rf_neighbor_map.get(nb_cell_sel, [])

    if cell_nbs:
        for nb in cell_nbs:
            cov_color = "#44cc44" if nb["coverage_match"] == "Yes" else "#ffaa22"
            st.markdown(f"""
            <div style="background:#101c30;border:1px solid #1e3a5a;border-radius:8px;
                        padding:10px 14px;margin:5px 0;font-size:0.84rem;color:#b0ccee;">
                <span style="font-weight:700;color:#63b3ed">🔗 {nb['cell_B']}</span>
                &nbsp;|&nbsp; Score: <b style="color:#63b3ed">{nb['neighbor_score']:.4f}</b>
                &nbsp;|&nbsp; In Beam: <b style="color:{cov_color}">{nb['coverage_match']}</b><br>
                Distance: <b>{nb['distance_m']:.0f} m</b> &nbsp;·&nbsp;
                Bearing: <b>{nb['bearing_deg']:.1f}°</b> &nbsp;·&nbsp;
                Az Diff: <b>{nb['az_diff_deg']:.1f}°</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No RF neighbors found for this cell (score below threshold).")


# ═══════════════════════════════════════════════════════════════
# TAB 5: Full Report
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### 📋 Full SON Analysis Report")

    display_df = report[[
        "Cell_Name", "GNB_ID", "Prediction", "Priority",
        "Mean_RSSI_dBm", "Std_dB", "Range_dB", "Peak_Drop_dB",
        "RF_Neighbor_Sim", "Top_RF_Neighbors",
    ]].copy()

    def color_pred(val):
        return f"color: {PRED_COLOR.get(val, '#fff')}"
    def color_priority(val):
        return f"color: {PRIORITY_COLOR.get(val, '#fff')}"

    styled = (
        display_df.style
        .map(color_pred,     subset=["Prediction"])
        .map(color_priority, subset=["Priority"])
        .format({
            "Mean_RSSI_dBm": "{:.1f}",
            "Std_dB":        "{:.1f}",
            "Range_dB":      "{:.1f}",
            "Peak_Drop_dB":  "{:.1f}",
            "RF_Neighbor_Sim": "{:.4f}",
        })
        .set_properties(**{"background-color": "#0d1b2e", "color": "#d0e0ff",
                           "border-color": "#1e3a5a"})
    )
    st.dataframe(styled, use_container_width=True, height=480)

    st.markdown("#### Recommended Actions by Cell")
    for _, row in report.iterrows():
        lbl   = row["Prediction"]
        color = PRED_COLOR[lbl]
        with st.expander(f"{row['Cell_Name']} — "
                         f"[{lbl}] | Priority: {row['Priority']}"):
            st.markdown(f"**Root Cause:** {row['Root_Cause']}")
            st.markdown(f"**Mean RSSI:** {row['Mean_RSSI_dBm']} dBm  "
                        f"| Std: {row['Std_dB']} dB  "
                        f"| Peak Drop: {row['Peak_Drop_dB']} dB")
            st.markdown("**Actions:**")
            for action in row["SON_Actions"].split(" | "):
                st.markdown(f"- {action}")
            st.markdown(f"**KPI Impact:** {row['KPI_Impact']}")

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Report (CSV)",
        data=csv, file_name="son_report_v3.csv", mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#444;font-size:0.76rem'>"
    "Auto SON Decision Engine v3.0 · Raw-dBm Classification · "
    "RF-Aware Neighbor Intelligence · "
    "Detect → Diagnose → Decide → Recommend · Built with Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
