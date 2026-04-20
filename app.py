"""
Early Mental Health Signal Detection — Streamlit Dashboard
For licensed mental health professionals and school counselors ONLY.

Privacy guarantees:
  • No input text is stored anywhere (session state only, cleared on close)
  • All inference runs locally — no external API calls
  • No logging of user text
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mental Health Signal Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ═══════════════════════════════════════════════════════════════
       QUIET SANCTUARY — Design System
       Mental Health Signal Detector · Counselor Edition
    ═══════════════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

    :root {
        /* Backgrounds */
        --bg-base:    #F7F3EE;
        --bg-surface: #FDFCFA;
        --bg-raised:  #F0EBE3;
        --bg-sidebar: #EFF4EC;
        --bg-hover:   #E8E2D9;

        /* Text */
        --text-primary:   #2C2016;
        --text-secondary: #5C4E3A;
        --text-muted:     #8A7A68;
        --text-disabled:  #C0B4A4;

        /* Sage green — trust, calm, nature, healing */
        --sage:        #4A7C59;
        --sage-mid:    #3D6B4A;
        --sage-light:  #6A9E78;
        --sage-subtle: rgba(74,124,89,0.08);
        --sage-glow:   rgba(74,124,89,0.15);

        /* Terracotta — warm human accent */
        --terra:        #C2714F;
        --terra-subtle: rgba(194,113,79,0.08);

        /* Risk levels — muted, never alarming */
        --risk-minimal:         #4A7C59;
        --risk-minimal-subtle:  rgba(74,124,89,0.10);
        --risk-low:             #7A6A30;
        --risk-low-subtle:      rgba(122,106,48,0.10);
        --risk-moderate:        #C2714F;
        --risk-moderate-subtle: rgba(194,113,79,0.10);
        --risk-high:            #A63D50;
        --risk-high-subtle:     rgba(166,61,80,0.10);

        /* Warning / disclaimer */
        --warn:        #8A6A1A;
        --warn-bg:     #FDF6E3;
        --warn-border: rgba(138,106,26,0.22);

        /* Privacy / info */
        --info:        #3A5F8A;
        --info-bg:     #EEF4FB;
        --info-border: rgba(58,95,138,0.20);

        /* Structure */
        --border:       #DDD6CC;
        --border-strong:#C8BFB4;
        --shadow-sm:    0 1px 4px rgba(44,32,22,0.06);
        --shadow-md:    0 4px 16px rgba(44,32,22,0.08);

        /* Fonts */
        --font-display: 'Lora', Georgia, serif;
        --font-body:    'Source Sans 3', system-ui, sans-serif;
        --font-mono:    'JetBrains Mono', 'Courier New', monospace;

        /* Radii */
        --r-sm:   6px;
        --r-md:   10px;
        --r-lg:   16px;
        --r-full: 9999px;

        /* Transitions */
        --t-fast: 140ms ease;
        --t-base: 240ms ease;
    }

    /* ── Global reset ─────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Warm text selection */
    ::selection { background: var(--sage-subtle); color: var(--sage); }

    /* Gentle scrollbar */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--border-strong);
        border-radius: var(--r-full);
    }

    /* Focus ring */
    *:focus-visible {
        outline: 2px solid var(--sage-light) !important;
        outline-offset: 3px;
        border-radius: var(--r-sm);
    }

    /* ── Headings ─────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: var(--font-display) !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
        line-height: 1.3;
    }
    h1, .stMarkdown h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

    /* ── Main content area ────────────────────────────────────── */
    .stMainBlockContainer, .main .block-container {
        background: var(--bg-base) !important;
        padding-top: 2rem !important;
        max-width: 1120px !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"], .css-1d391kg {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        font-family: var(--font-body) !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stSidebar"] h2 {
        font-family: var(--font-display) !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] h3 {
        font-family: var(--font-body) !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: var(--risk-high) !important;
    }

    /* Radio nav */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        padding: 6px 10px !important;
        border-radius: var(--r-md) !important;
        cursor: pointer;
        transition: background var(--t-fast);
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--sage-subtle) !important;
        color: var(--sage) !important;
    }

    /* Caption / version line */
    [data-testid="stSidebar"] .stCaption {
        font-family: var(--font-mono) !important;
        font-size: 11px !important;
        color: var(--text-disabled) !important;
    }

    /* ── Disclaimer box ───────────────────────────────────────── */
    .disclaimer-box {
        background: var(--warn-bg) !important;
        border: 1px solid var(--warn-border) !important;
        border-left: 3px solid var(--warn) !important;
        border-radius: var(--r-md) !important;
        padding: 12px 14px !important;
        font-size: 12.5px !important;
        color: var(--text-secondary) !important;
        line-height: 1.65 !important;
        margin-bottom: 10px !important;
    }
    .disclaimer-box strong { color: var(--warn) !important; }

    /* ── Privacy box ──────────────────────────────────────────── */
    .privacy-box {
        background: var(--info-bg) !important;
        border: 1px solid var(--info-border) !important;
        border-left: 3px solid var(--info) !important;
        border-radius: var(--r-md) !important;
        padding: 9px 12px !important;
        font-size: 12.5px !important;
        color: var(--info) !important;
        line-height: 1.6 !important;
        margin-bottom: 10px !important;
    }

    /* ── Textarea ─────────────────────────────────────────────── */
    textarea, .stTextArea textarea {
        background: var(--bg-surface) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: var(--r-lg) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
        padding: 14px 18px !important;
        box-shadow: var(--shadow-sm) !important;
        transition: border-color var(--t-base), box-shadow var(--t-base) !important;
        resize: vertical !important;
    }
    textarea:focus, .stTextArea textarea:focus {
        border-color: var(--sage-light) !important;
        box-shadow: 0 0 0 3px var(--sage-glow) !important;
        outline: none !important;
    }
    textarea::placeholder, .stTextArea textarea::placeholder {
        color: var(--text-disabled) !important;
        font-style: italic !important;
    }

    /* Textarea label */
    .stTextArea label {
        font-family: var(--font-body) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 6px !important;
    }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-body) !important;
        font-weight: 600 !important;
        font-size: 14.5px !important;
        border-radius: var(--r-md) !important;
        padding: 10px 26px !important;
        transition: all var(--t-base) !important;
        letter-spacing: 0.01em !important;
        cursor: pointer !important;
    }
    /* Primary — calm sage green, never alarming red */
    .stButton > button[kind="primary"] {
        background: var(--sage) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(74,124,89,0.22) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--sage-mid) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(74,124,89,0.28) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: scale(0.98) !important;
    }
    /* Secondary */
    .stButton > button[kind="secondary"] {
        background: var(--bg-surface) !important;
        color: var(--sage) !important;
        border: 1.5px solid var(--sage) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--sage-subtle) !important;
    }

    /* ── Metrics ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-lg) !important;
        padding: 14px 18px !important;
        box-shadow: var(--shadow-sm) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-body) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: 1.6rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* ── Alerts (success / warning / error / info) ────────────── */
    .stAlert {
        border-radius: var(--r-md) !important;
        font-family: var(--font-body) !important;
        font-size: 14px !important;
    }
    /* Success → sage-tinted */
    div[data-baseweb="notification"][kind="positive"],
    .stSuccess {
        background: var(--risk-minimal-subtle) !important;
        border-color: rgba(74,124,89,0.25) !important;
        color: var(--sage-mid) !important;
    }
    /* Warning → warm amber */
    div[data-baseweb="notification"][kind="warning"],
    .stWarning {
        background: var(--warn-bg) !important;
        border-color: var(--warn-border) !important;
        color: var(--warn) !important;
    }
    /* Error → muted rose, not harsh red */
    div[data-baseweb="notification"][kind="negative"],
    .stError {
        background: var(--risk-high-subtle) !important;
        border-color: rgba(166,61,80,0.2) !important;
        color: var(--risk-high) !important;
    }
    /* Info → soft blue */
    div[data-baseweb="notification"][kind="info"],
    .stInfo {
        background: var(--info-bg) !important;
        border-color: var(--info-border) !important;
        color: var(--info) !important;
    }

    /* ── Divider ──────────────────────────────────────────────── */
    hr, .stDivider {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.25rem 0 !important;
    }

    /* ── Dataframe / table ────────────────────────────────────── */
    .stDataFrame, [data-testid="stDataFrameResizable"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
    }

    /* ── File uploader ────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--bg-surface) !important;
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: var(--r-lg) !important;
        padding: 20px !important;
        transition: border-color var(--t-base), background var(--t-base) !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--sage-light) !important;
        background: var(--sage-subtle) !important;
    }
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] label {
        font-family: var(--font-body) !important;
        font-size: 14px !important;
        color: var(--text-secondary) !important;
    }

    /* ── Spinner ──────────────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: var(--sage) !important;
    }

    /* ── Plotly chart containers ──────────────────────────────── */
    [data-testid="stPlotlyChart"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-lg) !important;
        padding: 8px !important;
        box-shadow: var(--shadow-sm) !important;
        overflow: hidden !important;
    }

    /* ── Expander ─────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        font-family: var(--font-body) !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }
    .streamlit-expanderContent {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--r-md) var(--r-md) !important;
    }

    /* ── Code blocks (CSV format example) ────────────────────── */
    .stCode, code, pre {
        font-family: var(--font-mono) !important;
        font-size: 12.5px !important;
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Caption text ─────────────────────────────────────────── */
    .stCaption, small {
        font-size: 12px !important;
        color: var(--text-muted) !important;
        font-style: italic !important;
    }

    /* ── Existing component classes (kept intact, restyled) ───── */

    /* Risk level inline color spans */
    .risk-HIGH     { color: var(--risk-high)     !important; font-weight: 700; }
    .risk-MODERATE { color: var(--risk-moderate) !important; font-weight: 700; }
    .risk-LOW      { color: var(--risk-low)      !important; font-weight: 700; }
    .risk-MINIMAL  { color: var(--risk-minimal)  !important; font-weight: 700; }

    /* Resource cards */
    .resource-card {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-md) !important;
        padding: 10px 14px !important;
        margin: 5px 0 !important;
        font-size: 13.5px !important;
        font-family: var(--font-body) !important;
        box-shadow: var(--shadow-sm) !important;
        transition: box-shadow var(--t-fast) !important;
    }
    .resource-card:hover { box-shadow: var(--shadow-md) !important; }
    .resource-card strong { color: var(--text-primary) !important; font-weight: 600 !important; }
    .resource-card span   { color: var(--text-muted)    !important; }

    /* Feature chips */
    .feature-chip {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--r-full) !important;
        padding: 3px 12px !important;
        font-size: 12px !important;
        font-family: var(--font-mono) !important;
        color: var(--text-secondary) !important;
        margin: 3px !important;
        display: inline-block !important;
    }

    /* ── Phrase highlight container ───────────────────────────── */
    /* The yellow/green inline marks in build_highlighted_html() are
       injected with explicit inline styles — we soften those colors
       using CSS custom properties without overriding the style attr  */
    mark[style*="FFD700"] {
        background: rgba(212,132,10,0.18) !important;
        color: var(--text-primary) !important;
        border-radius: 3px !important;
        padding: 1px 3px !important;
        font-weight: 500 !important;
    }
    mark[style*="90EE90"] {
        background: rgba(74,124,89,0.15) !important;
        color: var(--text-primary) !important;
        border-radius: 3px !important;
        padding: 1px 3px !important;
        font-weight: 500 !important;
    }

    /* ── Entrance animations (gentle, unhurried) ──────────────── */
    @keyframes quietFadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    .resource-card,
    [data-testid="stMetric"],
    [data-testid="stPlotlyChart"] {
        animation: quietFadeIn 0.45s ease-out both;
    }

    /* ── Header — minimal and clean; sidebar toggle KEPT intact ── */
    header[data-testid="stHeader"] {
        background: var(--bg-base) !important;
        border-bottom: 1px solid var(--border) !important;
        box-shadow: none !important;
    }

    /* Hide only the noisy right-side toolbar (Deploy, share, ⋮ menu) */
    [data-testid="stToolbarActions"],
    .stDeployButton,
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    /* Sidebar toggle button in the header — styled to palette */
    header button,
    header [data-testid="stSidebarNavToggleButton"] {
        color: var(--text-secondary) !important;
        border-radius: var(--r-sm) !important;
        transition: background var(--t-fast), color var(--t-fast) !important;
    }
    header button:hover {
        background: var(--sage-subtle) !important;
        color: var(--sage) !important;
    }

    /* Decoration bar — sage gradient instead of Streamlit red */
    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--sage), var(--sage-light)) !important;
        height: 2px !important;
        opacity: 0.75 !important;
    }

    /* Collapsed sidebar strip — the thin edge when sidebar is closed */
    [data-testid="collapsedControl"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border) !important;
        color: var(--sage) !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: var(--sage-subtle) !important;
    }

    /* Fix Material Symbol icon rendering in sidebar toggle */
    .material-symbols-rounded,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="collapsedControl"] span {
        font-family: 'Material Symbols Rounded' !important;
        font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24 !important;
        font-size: 20px !important;
        color: var(--sage) !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
    }

    /* ── Fix inline word-count color spans (Python sets color names) */
    /* "37 words" uses style="color:orange" — remap to palette */
    span[style*="color:orange"],
    span[style*="color: orange"] {
        color: var(--terra) !important;       /* terracotta, not vivid orange */
    }
    span[style*="color:green"],
    span[style*="color: green"] {
        color: var(--sage) !important;        /* sage, not vivid green */
    }

    /* ── Sidebar bullet list (crisis resources) ───────────────── */
    [data-testid="stSidebar"] ul {
        list-style: none !important;
        padding-left: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] li {
        font-size: 13px !important;
        color: var(--text-secondary) !important;
        padding: 3px 0 !important;
        line-height: 1.55 !important;
        border-bottom: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] li:last-child {
        border-bottom: none !important;
    }
    [data-testid="stSidebar"] li strong {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] li em {
        color: var(--text-muted) !important;
        font-style: normal !important;
        font-size: 12px !important;
    }
    /* Sidebar link */
    [data-testid="stSidebar"] a {
        color: var(--sage) !important;
        font-size: 12.5px !important;
        text-decoration: underline !important;
        text-underline-offset: 2px !important;
    }

    /* ── Responsive ───────────────────────────────────────────── */
    @media (max-width: 768px) {
        .stMainBlockContainer { padding: 1rem !important; }
        h1 { font-size: 1.4rem !important; }
        textarea { font-size: 14px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Persistent disclaimer (shown on every page)
# ---------------------------------------------------------------------------

DISCLAIMER_HTML = """
<div class="disclaimer-box">
<strong>⚠️ PROFESSIONAL TOOL — NOT FOR SELF-DIAGNOSIS</strong><br>
This tool is designed for <em>licensed mental health professionals, school counselors, and clinicians only</em>.
It is not a diagnostic instrument. Results must be interpreted by a qualified professional in context.
<br><br>
<strong>🔒 Privacy:</strong> All text is processed <em>locally on this device</em>.
No input is stored, logged, or transmitted to any server. Session data is cleared when you close the browser tab.
</div>
"""

PRIVACY_NOTE_HTML = """
<div class="privacy-box">
🔒 <strong>Local processing only</strong> — no text leaves this device.
</div>
"""

# ---------------------------------------------------------------------------
# Model loader (cached for performance, NOT caching text)
# ---------------------------------------------------------------------------

MODEL_PATH = "models/saved"

@st.cache_resource(show_spinner="Loading model…")
def load_analyzer():
    """Load model once per session. Model weights cached, never text."""
    if not os.path.isdir(MODEL_PATH):
        return None
    try:
        from pipeline.analyze import MentalHealthAnalyzer
        return MentalHealthAnalyzer(model_path=MODEL_PATH)
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 Mental Health Signal Detector")
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Counselor View", "Trend Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🆘 Crisis Resources")
    st.markdown("""
- **iCall** — 9152987821 *Mon–Sat, 8am–10pm*
- **Vandrevala** — 1860-2662-345 *24/7*
- **Snehi** — 044-24640050 *24/7*
- **NIMHANS** — 080-46110007
- **112** — Emergency services

[More resources →](https://icallhelpline.org)
""")
    st.markdown("---")
    st.caption("v1.0 · Runs entirely offline · No data stored")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _risk_color(level: str) -> str:
    return {"HIGH": "#dc3545", "MODERATE": "#fd7e14", "LOW": "#ffc107", "MINIMAL": "#28a745"}.get(level, "#6c757d")


def _build_score_chart(scores: dict) -> go.Figure:
    labels = list(scores.keys())
    values = [scores[l] * 100 for l in labels]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 105], title="Probability (%)"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=40, t=20, b=20),
        height=200,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef")
    return fig


def _model_unavailable_msg():
    st.error(
        "**Model not found.** Please train the model first:\n\n"
        "```bash\n"
        "python -m models.train --output-dir models/saved --epochs 5\n"
        "```\n\n"
        "See `README.md` for dataset setup instructions."
    )


# ---------------------------------------------------------------------------
# PAGE 1 — Counselor View
# ---------------------------------------------------------------------------

if page == "Counselor View":
    st.title("Counselor Analysis View")
    st.markdown(PRIVACY_NOTE_HTML, unsafe_allow_html=True)

    st.markdown(
        "Paste a **de-identified** journal entry or chat message below. "
        "Remove all names, dates, locations, and other identifiers before analysis."
    )

    text_input = st.text_area(
        "Text to analyze",
        height=180,
        placeholder="Paste anonymized text here (50–2 000 words)…",
        help="All processing happens locally. Text is never stored.",
        key="counselor_input",
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    with col_info:
        word_count = len(text_input.split()) if text_input.strip() else 0
        if word_count > 0:
            color = "green" if 50 <= word_count <= 2000 else "orange"
            st.markdown(f"<span style='color:{color}'>{word_count} words</span>", unsafe_allow_html=True)
            if word_count < 50:
                st.caption("Tip: at least 50 words gives more reliable results.")

    if analyze_btn:
        text = text_input.strip()
        if len(text.split()) < 10:
            st.warning("Please enter at least 10 words.")
            st.stop()

        analyzer = load_analyzer()
        if analyzer is None:
            _model_unavailable_msg()
            st.stop()
        if isinstance(analyzer, str):
            st.error(f"Model load error: {analyzer}")
            st.stop()

        with st.spinner("Analyzing… (LIME phrase detection may take ~10 s)"):
            result = analyzer.analyze(text, num_lime_samples=50)

        # ── Risk Level Banner ──────────────────────────────────────────
        level  = result["risk_level"]
        primary = result["primary_label"]
        color  = _risk_color(level)

        st.markdown("---")
        st.markdown(
            f"<h3>Risk Level: <span style='color:{color}'>{level}</span> "
            f"<span style='font-size:0.7em;color:#6c757d'>primary signal: {primary}</span></h3>",
            unsafe_allow_html=True,
        )

        # ── Score Breakdown ────────────────────────────────────────────
        col_chart, col_features = st.columns([3, 2])

        with col_chart:
            st.subheader("Score Breakdown")
            fig = _build_score_chart(result["scores"])
            st.plotly_chart(fig, use_container_width=True)

        with col_features:
            st.subheader("Top Linguistic Signals")
            for feat in result["top_features"]:
                st.markdown(
                    f"""<div class="resource-card">
                    <strong>{feat['feature']}</strong><br>
                    <span style="color:#6c757d">{feat['description']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Highlighted Text ───────────────────────────────────────────
        from pipeline.analyze import build_highlighted_html

        st.subheader("Phrase Analysis")
        st.caption(
            "🟡 Yellow = language associated with detected risk signal  "
            "🟢 Green = language associated with resilience/safety"
        )
        html = build_highlighted_html(text, result["word_weights"])
        st.markdown(
            f'<div style="background:#fafafa;border:1px solid #dee2e6;border-radius:6px;'
            f'padding:14px;max-height:300px;overflow-y:auto">{html}</div>',
            unsafe_allow_html=True,
        )

        # ── Recommended Resources ──────────────────────────────────────
        st.subheader("Recommended Resources")
        for res in result["resources"]:
            st.markdown(
                f'<div class="resource-card"><strong>{res["name"]}</strong><br>'
                f'<span style="color:#555">{res["detail"]}</span></div>',
                unsafe_allow_html=True,
            )

        # ── Counselor Notes Prompt ─────────────────────────────────────
        st.markdown("---")
        st.info(
            "**Reminder:** This analysis is a decision-support tool only. "
            "Clinical judgement, direct conversation, and professional guidelines "
            "should always guide intervention decisions."
        )

        # Clear from memory after display — text lives only in the local variable
        del text


# ---------------------------------------------------------------------------
# PAGE 2 — Trend Analysis
# ---------------------------------------------------------------------------

elif page == "Trend Analysis":
    st.title("Trend Analysis")
    st.markdown(PRIVACY_NOTE_HTML, unsafe_allow_html=True)

    st.markdown(
        "Upload a CSV of **anonymized** entries to visualise risk trends over time. "
        "Required columns: **`date`** (YYYY-MM-DD) and **`entry`** (text)."
    )

    with st.expander("CSV format example"):
        st.code(
            "date,entry\n"
            "2024-01-10,\"Today I felt really anxious about my exam.\"\n"
            "2024-01-17,\"Things are a bit better this week.\"\n",
            language="csv",
        )

    uploaded = st.file_uploader("Upload CSV", type="csv", help="File is read in-memory only and never saved.")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "date" not in df.columns or "entry" not in df.columns:
            st.error("CSV must have **date** and **entry** columns.")
            st.stop()

        df = df.dropna(subset=["date", "entry"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if df.empty:
            st.warning("No valid rows found after parsing.")
            st.stop()

        st.success(f"{len(df)} entries loaded — dates: {df['date'].min().date()} → {df['date'].max().date()}")

        analyzer = load_analyzer()
        if analyzer is None:
            _model_unavailable_msg()
            st.stop()
        if isinstance(analyzer, str):
            st.error(f"Model load error: {analyzer}")
            st.stop()

        with st.spinner(f"Scoring {len(df)} entries…"):
            texts = df["entry"].tolist()
            batch_results = analyzer.analyze_batch(texts)

        from models.model import LABEL_NAMES

        for label in LABEL_NAMES:
            df[label] = [r["scores"][label] for r in batch_results]
        df["risk_level"] = [r["risk_level"] for r in batch_results]

        # ── Risk Trend Line Chart ───────────────────────────────────────
        st.subheader("Risk Score Trends Over Time")

        fig = go.Figure()
        palette = {"depression": "#4e79a7", "anxiety": "#f28e2b", "crisis": "#e15759", "neutral": "#76b7b2"}
        visible = {"depression": True, "anxiety": True, "crisis": True, "neutral": "legendonly"}

        for label in LABEL_NAMES:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[label],
                name=label.capitalize(),
                mode="lines+markers",
                line=dict(color=palette[label], width=2),
                marker=dict(size=6),
                visible=visible[label],
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            legend_title="Category",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e9ecef")
        fig.update_yaxes(showgrid=True, gridcolor="#e9ecef")
        st.plotly_chart(fig, use_container_width=True)

        # ── Risk Level Timeline ─────────────────────────────────────────
        st.subheader("Risk Level per Entry")

        level_color = {"HIGH": "#dc3545", "MODERATE": "#fd7e14", "LOW": "#ffc107", "MINIMAL": "#28a745"}
        df_disp = df[["date", "risk_level"]].copy()
        df_disp["color"] = df_disp["risk_level"].map(level_color)

        fig2 = px.scatter(
            df_disp,
            x="date",
            y=[1] * len(df_disp),
            color="risk_level",
            color_discrete_map=level_color,
            hover_data={"date": True, "risk_level": True},
            labels={"y": ""},
            height=120,
        )
        fig2.update_traces(marker=dict(size=14, symbol="square"))
        fig2.update_yaxes(visible=False)
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

        # ── Summary Statistics ──────────────────────────────────────────
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        level_counts = df["risk_level"].value_counts()

        col1.metric("HIGH risk entries",     int(level_counts.get("HIGH", 0)))
        col2.metric("MODERATE risk entries", int(level_counts.get("MODERATE", 0)))
        col3.metric("LOW risk entries",      int(level_counts.get("LOW", 0)))
        col4.metric("MINIMAL risk entries",  int(level_counts.get("MINIMAL", 0)))

        # Average scores table
        avg_scores = df[LABEL_NAMES].mean().rename("Mean Score").round(3)
        st.markdown("**Average risk scores across all entries:**")
        st.dataframe(avg_scores.to_frame().T, use_container_width=True)

        # Crisis spike warning
        high_rows = df[df["risk_level"] == "HIGH"]
        if not high_rows.empty:
            st.warning(
                f"**{len(high_rows)} HIGH-risk {'entry' if len(high_rows)==1 else 'entries'} detected.** "
                "These warrant immediate professional follow-up."
            )

        # Clear uploaded data from memory
        del df, batch_results, texts

    st.markdown("---")
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
