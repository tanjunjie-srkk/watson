"""
Watson Document Intelligence Platform
Streamlit UI for OCR, Extraction & Bank Matching Demo
"""

import streamlit as st
import json
import sys
import base64
import tempfile
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdf_to_images import pdf_to_images
from ocr_agent import ocr_images_with_chat_model, ocr_image_with_chat_model, _maybe_parse_json
from orchestrator import run as orchestrator_run, AGENT_REGISTRY
from agents.classifier import classify_document

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Watson Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Watsons Brand Theme ────────────────────────────────────────
# Palette:  Teal #006770 · #00A0AF · #00a3b2 · #70c9d2 · #57e2c8
#           Red  #EE2D25 · White #FFFFFF
st.markdown("""
<style>
    /* ═══ GLOBAL FOUNDATION ═══ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }

    /* ═══ MAIN HEADER BAR ═══ */
    .main-header {
        background: linear-gradient(135deg, #005a62 0%, #006770 30%, #00A0AF 100%);
        padding: 1.4rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
        border-bottom: 3px solid #57e2c8;
        box-shadow: 0 4px 16px rgba(0,103,112,0.18);
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0; font-weight: 800; letter-spacing: -0.01em; }
    .main-header p { color: #b2e8ec; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    .pipeline-step {
        background: #f0fafb; border-left: 4px solid #00A0AF;
        padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 0.8rem;
    }
    .confidence-high { color: #00A0AF; font-weight: 600; }
    .confidence-medium { color: #d68000; font-weight: 600; }
    .confidence-low { color: #EE2D25; font-weight: 600; }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #004d55 0%, #006770 40%, #007a85 100%) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] .stCaption p { color: #b2e8ec !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    /* Sidebar selectbox / radio / widget labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #d0f0f4 !important;
    }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: #e0f7f9 !important;
    }
    /* Radio button active state */
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
        background: rgba(87,226,200,0.15) !important;
        border-radius: 6px;
    }

    .sidebar-logo {
        text-align: center; padding: 1.2rem 0 0.6rem 0;
    }
    .sidebar-logo-icon {
        font-size: 2.4rem; display: block; margin-bottom: 0.3rem;
    }
    .sidebar-logo-text {
        font-size: 1.2rem; font-weight: 800; color: #ffffff;
        letter-spacing: 0.02em;
    }
    .sidebar-logo-sub {
        font-size: 0.72rem; color: #70c9d2; letter-spacing: 0.06em;
        text-transform: uppercase; margin-top: 2px;
    }
    .sidebar-section {
        font-size: 0.68rem; font-weight: 700; color: #57e2c8;
        text-transform: uppercase; letter-spacing: 0.1em;
        padding: 0.8rem 0 0.4rem 0.5rem;
    }
    .sidebar-about-card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.3rem;
    }
    .sidebar-about-title {
        font-size: 0.7rem; font-weight: 600; color: #70c9d2;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;
    }
    .sidebar-about-item {
        display: flex; align-items: flex-start; gap: 0.5rem;
        padding: 0.35rem 0; font-size: 0.8rem; color: #b2e8ec;
    }
    .sidebar-about-item strong { color: #ffffff; }
    .sidebar-badge {
        display: inline-block; font-size: 0.62rem; font-weight: 700;
        padding: 2px 10px; border-radius: 20px;
        background: rgba(87,226,200,0.2); color: #57e2c8;
        letter-spacing: 0.03em;
    }
    .sidebar-footer {
        text-align: center; padding: 0.5rem 0; font-size: 0.72rem; color: #70c9d2;
    }

    /* ═══ STREAMLIT WIDGET OVERRIDES (Watsons Teal) ═══ */
    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #006770, #00A0AF) !important;
        border: none !important; color: white !important; font-weight: 600;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0,103,112,0.18) !important;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #005a62, #008a96) !important;
        box-shadow: 0 4px 12px rgba(0,103,112,0.3) !important;
    }
    /* Secondary buttons */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        border: 1.5px solid #00A0AF !important; color: #006770 !important;
        font-weight: 600; border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #e0f7f9 !important; border-color: #006770 !important;
    }
    /* Default buttons */
    .stButton > button {
        border: 1.5px solid #d0e8eb !important; color: #006770 !important;
        font-weight: 500; border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #e0f7f9 !important; border-color: #00A0AF !important;
        color: #005a62 !important;
    }
    /* Download buttons */
    .stDownloadButton > button {
        border: 1.5px solid #00A0AF !important; color: #006770 !important;
        font-weight: 600; border-radius: 8px !important;
    }
    .stDownloadButton > button:hover {
        background: #e0f7f9 !important;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0 !important; font-weight: 600;
        color: #006770 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #e0f7f9 !important; color: #006770 !important;
        border-bottom: 3px solid #00A0AF !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] { color: #006770 !important; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #4a6d71 !important; }
    /* Expanders */
    details[data-testid="stExpander"] summary {
        color: #006770 !important; font-weight: 600;
    }
    /* Selectbox / multiselect borders */
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        border-color: #b2e8ec !important; border-radius: 8px !important;
    }
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stMultiSelect [data-baseweb="select"] > div:focus-within {
        border-color: #00A0AF !important;
        box-shadow: 0 0 0 1px #00A0AF !important;
    }
    /* Progress bar */
    .stProgress > div > div > div { background: #00A0AF !important; }
    /* Dividers */
    hr { border-color: #d0e8eb !important; }
    /* File uploader */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #70c9d2 !important; border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #00A0AF !important; background: #f0fafb !important;
    }
    /* Toggle */
    .stToggle label span[data-checked="true"] {
        background-color: #00A0AF !important;
    }
    /* Text input */
    .stTextInput input { border-color: #b2e8ec !important; border-radius: 8px !important; }
    .stTextInput input:focus { border-color: #00A0AF !important; box-shadow: 0 0 0 1px #00A0AF !important; }
    /* Status widget */
    [data-testid="stStatusWidget"] { border-left: 4px solid #00A0AF !important; }

    /* ═══ INFO / SUCCESS / WARNING / ERROR BANNERS ═══ */
    .stAlert [data-testid="stAlertContentInfo"] {
        background: #e0f7f9 !important; border-left-color: #00A0AF !important;
    }

    /* ═══ GENERAL CARDS ═══ */
    .info-card {
        background: #fff; border: 1px solid #d0e8eb; border-radius: 12px;
        padding: 1rem; margin-bottom: 0.5rem; box-shadow: 0 1px 4px rgba(0,103,112,0.06);
    }
    .info-card .card-title { font-weight: 700; color: #006770; margin-bottom: 0.3rem; }
    .info-card .card-subtitle { font-size: 0.8rem; color: #5a8a8f; }

    /* ═══ REPORT DOCUMENT CARDS ═══ */
    .doc-card {
        background: #fff; border: 1px solid #d0e8eb; border-radius: 12px;
        padding: 0; margin-bottom: 0.75rem; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,103,112,0.06); transition: all 0.2s;
    }
    .doc-card:hover { box-shadow: 0 4px 14px rgba(0,103,112,0.12); border-color: #70c9d2; }
    .doc-card-header {
        display: flex; align-items: center; padding: 0.8rem 1.2rem;
        gap: 1rem; cursor: pointer;
    }
    .doc-card-num {
        font-weight: 700; font-size: 0.85rem; color: #5a8a8f;
        min-width: 32px; text-align: center;
    }
    .doc-card-type {
        font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; text-transform: uppercase; letter-spacing: 0.03em;
    }
    .doc-card-company { font-weight: 600; color: #006770; font-size: 0.95rem; flex: 1; }
    .doc-card-detail { color: #4a6d71; font-size: 0.85rem; }
    .doc-card-amount { font-weight: 700; color: #006770; font-size: 1rem; white-space: nowrap; }
    .doc-card-match {
        font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;
        font-weight: 600; white-space: nowrap;
    }
    .doc-field-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 0.6rem; padding: 0.8rem 1.2rem;
    }
    .doc-field { padding: 0.4rem 0; }
    .doc-field-label { font-size: 0.7rem; color: #70a0a5; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
    .doc-field-value { font-size: 0.88rem; color: #1a3d42; font-weight: 500; word-break: break-word; }
    .doc-status-verified { border-left: 4px solid #00A0AF; }
    .doc-status-rejected { border-left: 4px solid #EE2D25; opacity: 0.6; }
    .doc-status-pending { border-left: 4px solid #d0e8eb; }
    .doc-card-status { font-size: 0.8rem; white-space: nowrap; color: #5a8a8f; }

    /* ═══ LINE-ITEM MATCHING PAGE — AP Reconciliation ═══ */
    .lim-header {
        background: linear-gradient(135deg, #004d55 0%, #006770 50%, #00A0AF 100%);
        padding: 1.6rem 2.2rem; border-radius: 12px; margin-bottom: 1.2rem;
        display: flex; align-items: center; gap: 1.2rem;
        border-bottom: 3px solid #57e2c8;
        box-shadow: 0 4px 16px rgba(0,103,112,0.18);
    }
    .lim-header-icon { font-size: 2.2rem; }
    .lim-header h2 { color: #fff; margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.01em; }
    .lim-header p { color: #b2e8ec; margin: 0.2rem 0 0 0; font-size: 0.85rem; }

    /* KPI cards */
    .lim-kpi-card {
        background: #fff; border: 1px solid #d0e8eb; border-radius: 12px;
        padding: 0.9rem 1rem; text-align: center;
        box-shadow: 0 1px 4px rgba(0,103,112,0.06); position: relative;
    }
    .lim-kpi-card .kpi-value { font-size: 1.45rem; font-weight: 800; margin-bottom: 0.1rem; line-height: 1.2; }
    .lim-kpi-card .kpi-label {
        font-size: 0.68rem; color: #5a8a8f; text-transform: uppercase;
        letter-spacing: 0.06em; font-weight: 600;
    }
    .lim-kpi-card .kpi-tooltip {
        font-size: 0.65rem; color: #8ab0b5; margin-top: 0.2rem;
        font-style: italic;
    }
    .lim-kpi-card .kpi-border-top { position: absolute; top: 0; left: 10%; right: 10%; height: 3px; border-radius: 0 0 3px 3px; }
    .lim-kpi-teal .kpi-value { color: #006770; }
    .lim-kpi-teal .kpi-border-top { background: #006770; }
    .lim-kpi-blue .kpi-value { color: #00A0AF; }
    .lim-kpi-blue .kpi-border-top { background: #00A0AF; }
    .lim-kpi-green .kpi-value { color: #00856e; }
    .lim-kpi-green .kpi-border-top { background: #00856e; }
    .lim-kpi-red .kpi-value { color: #EE2D25; }
    .lim-kpi-red .kpi-border-top { background: #EE2D25; }
    .lim-kpi-amber .kpi-value { color: #d68000; }
    .lim-kpi-amber .kpi-border-top { background: #d68000; }
    .lim-kpi-navy .kpi-value { color: #004d55; }
    .lim-kpi-navy .kpi-border-top { background: #004d55; }

    .lim-section-title {
        font-size: 0.95rem; font-weight: 700; color: #006770;
        padding: 0.5rem 0 0.25rem 0; display: flex; align-items: center; gap: 0.5rem;
    }

    /* Tables */
    .lim-table {
        width: 100%; border-collapse: separate; border-spacing: 0;
        font-size: 0.84rem; border-radius: 10px; overflow: hidden;
        border: 1px solid #b2dfe4;
    }
    .lim-table thead th {
        background: #006770; color: #fff; padding: 0.55rem 0.8rem;
        font-weight: 700; font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.04em; text-align: left;
    }
    .lim-table tbody td {
        padding: 0.5rem 0.8rem; border-bottom: 1px solid #e0f0f2;
        color: #1a3d42; font-size: 0.84rem;
    }
    .lim-table tbody tr:last-child td { border-bottom: none; }
    .lim-table tbody tr:hover { background: #f0fafb; }
    .lim-table .amount-col { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
    .lim-table .neg-amount { color: #EE2D25; }
    .lim-table-ledger thead th { background: #00856e; }
    .lim-table-result thead th { background: #004d55; }
    .lim-table-result tbody td { font-size: 0.83rem; }

    /* Status badges */
    .lim-badge {
        display: inline-flex; align-items: center; gap: 0.3rem;
        padding: 2px 10px; border-radius: 20px; font-weight: 700;
        font-size: 0.73rem; white-space: nowrap;
    }
    .lim-badge-match { background: #e0f7f0; color: #00856e; }
    .lim-badge-mismatch { background: #fde8e7; color: #EE2D25; }
    .lim-badge-missing-ledger { background: #fff3e0; color: #d68000; }
    .lim-badge-missing-soa { background: #e0f7f9; color: #00A0AF; }
    .lim-badge-investigating { background: #e0f7f9; color: #00A0AF; }
    .lim-badge-resolved { background: #e0f7f0; color: #006770; }
    .lim-badge-approved { background: #e0f7f0; color: #004d55; border: 1px solid #70c9d2; }

    /* Summary / reconciliation bar */
    .lim-recon-bar {
        background: #fff; border: 1px solid #d0e8eb; border-radius: 12px;
        padding: 1rem 1.5rem; margin-top: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,103,112,0.06);
    }
    .lim-recon-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.3rem 0; font-size: 0.88rem;
    }
    .lim-recon-row .recon-label { color: #4a6d71; font-weight: 500; }
    .lim-recon-row .recon-value { font-weight: 700; color: #006770; font-variant-numeric: tabular-nums; }
    .lim-recon-divider { border-bottom: 1px dashed #b2dfe4; margin: 0.3rem 0; }
    .lim-progress-track {
        background: #e0f0f2; border-radius: 6px; height: 10px; width: 100%;
        overflow: hidden; margin-top: 0.4rem;
    }
    .lim-progress-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }

    /* Variance rows */
    .lim-variance-row {
        background: #fff8e6; border-left: 4px solid #d68000;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0;
        margin-bottom: 0.4rem; font-size: 0.84rem;
    }
    /* Filter bar */
    .lim-filter-bar {
        background: #f0fafb; border: 1px solid #d0e8eb; border-radius: 12px;
        padding: 0.8rem 1.2rem; margin-bottom: 1rem;
    }
    /* Drill-down panel */
    .lim-drill-panel {
        background: #fff; border: 1px solid #b2dfe4; border-radius: 12px;
        padding: 1.2rem 1.5rem; margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,103,112,0.08);
    }
    .lim-drill-field { margin-bottom: 0.5rem; }
    .lim-drill-field .dl-label {
        font-size: 0.68rem; color: #5a8a8f; text-transform: uppercase;
        letter-spacing: 0.05em; font-weight: 600; margin-bottom: 1px;
    }
    .lim-drill-field .dl-value { font-size: 0.92rem; color: #006770; font-weight: 600; }
    .lim-drill-field .dl-value.dl-highlight { color: #EE2D25; background: #fde8e7; padding: 1px 6px; border-radius: 4px; }
    .lim-drill-field .dl-value.dl-ok { color: #00856e; }

    .lim-comment-box {
        width: 100%; border: 1px solid #b2dfe4; border-radius: 8px;
        padding: 0.5rem 0.7rem; font-size: 0.84rem; resize: vertical;
        min-height: 60px; font-family: inherit;
    }
    .lim-comment-box:focus { border-color: #00A0AF; outline: none; box-shadow: 0 0 0 2px rgba(0,160,175,0.15); }

    /* ═══ DATAFRAME THEMING ═══ */
    .stDataFrame { border-radius: 10px !important; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPER FUNCTIONS (must be defined before page logic calls them)
# ═══════════════════════════════════════════════════════════════════════════

def display_confidence_bar(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.95:
        return f"🟢 {pct}%"
    elif score >= 0.85:
        return f"🟡 {pct}%"
    else:
        return f"🔴 {pct}%"


def load_json_file(path: Path) -> dict | list | str:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Error: {e}"


def display_processing_file_preview(file_path: Path):
    """Preview a source document file from docs/database."""
    if not file_path.exists() or not file_path.is_file():
        st.error("Selected document file was not found.")
        return

    suffix = file_path.suffix.lower()
    st.caption(f"File: {file_path.name}")

    if suffix == ".pdf":
        pdf_bytes = file_path.read_bytes()
        # Render PDF pages as images (avoids iframe/CSP blocking on Streamlit Cloud)
        try:
            import tempfile, fitz  # noqa: E401
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages = len(doc)
            st.markdown(
                f'<div style="color:#5a8a8f;font-size:0.82rem;margin-bottom:0.4rem;">'
                f'📄 {n_pages} page{"s" if n_pages != 1 else ""}</div>',
                unsafe_allow_html=True,
            )
            for page_idx in range(n_pages):
                pix = doc[page_idx].get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                st.image(img_bytes, caption=f"Page {page_idx + 1} of {n_pages}", use_container_width=True)
            doc.close()
        except Exception as img_err:
            st.warning(f"Could not render PDF pages as images: {img_err}")
            st.info("Use the download button below to view the PDF.")
        st.download_button(
            "⬇️ Download PDF",
            data=pdf_bytes,
            file_name=file_path.name,
            mime="application/pdf",
        )
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        st.image(str(file_path), use_container_width=True)
        st.download_button(
            "⬇️ Download Image",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime="application/octet-stream",
        )
    else:
        st.info("Preview is not available for this file type. You can download the file below.")
        st.download_button(
            "⬇️ Download File",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime="application/octet-stream",
        )


def display_ocr_result(data: dict):
    """Display OCR output with confidence scoring and section breakdown."""
    pages = []
    if "model_output" in data and isinstance(data["model_output"], dict):
        pages = data["model_output"].get("pages", [])
    elif "results" in data:
        for result in data["results"]:
            if isinstance(result, dict):
                mo = result.get("model_output", {})
                if isinstance(mo, dict) and "pages" in mo:
                    pages.extend(mo["pages"])
                elif isinstance(mo, dict) and "pages" not in mo:
                    pass  # skip non-page outputs
    elif "pages" in data:
        pages = data["pages"]

    if not pages:
        st.warning("No structured OCR pages found. Showing raw JSON.")
        st.json(data)
        return

    # Summary metrics
    total_sections = sum(len(p.get("sections", [])) for p in pages)
    avg_confidence = 0.0
    conf_count = 0
    for p in pages:
        for s in p.get("sections", []):
            c = s.get("confidence", 0)
            if c > 0:
                avg_confidence += c
                conf_count += 1
    avg_confidence = avg_confidence / conf_count if conf_count > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Pages", len(pages))
    c2.metric("📝 Sections", total_sections)
    c3.metric("🎯 Avg Confidence", f"{avg_confidence:.0%}")
    st.divider()

    for page_data in pages:
        page_num = page_data.get("page_number", "?")
        file_name = page_data.get("file_name", "")
        sections = page_data.get("sections", [])

        with st.expander(f"📄 Page {page_num} — {file_name} ({len(sections)} sections)", expanded=(page_num == 1 or page_num == "1")):
            if not sections:
                st.caption("No sections.")
                continue

            type_counts: dict[str, int] = {}
            for s in sections:
                t = s.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            st.markdown("**Section types:** " + " ".join(f"`{t}` ×{c}" for t, c in sorted(type_counts.items(), key=lambda x: -x[1])))

            icon_map = {
                "header": "📌", "address": "📍", "key_value": "🔑",
                "table_header": "📊", "table_row": "📋", "subtotal": "💰",
                "paragraph": "📝", "footer": "📎", "signature": "✍️", "empty": "⬜",
            }

            for i, s in enumerate(sections):
                sec_type = s.get("type", "unknown")
                content = s.get("content", "")
                confidence = s.get("confidence", 0)
                icon = icon_map.get(sec_type, "📄")

                with st.container():
                    col_c, col_conf = st.columns([5, 1])
                    with col_c:
                        st.markdown(f"**{icon} {sec_type.upper()}**")
                        if sec_type in ("table_header", "table_row"):
                            st.code(content, language=None)
                        else:
                            st.text(content[:500])
                    with col_conf:
                        st.markdown(f"**{display_confidence_bar(confidence)}**")
                    if i < len(sections) - 1:
                        st.markdown("---")


def display_extraction_result(data: dict, doc_type: str = "Unknown"):
    """Display extraction result in structured, professional format."""
    if not isinstance(data, dict):
        st.json(data)
        return

    actual_type = data.get("document_type", doc_type)
    st.markdown(f"**Document Type:** `{actual_type}`")
    currency_code = str(data.get("currency") or "").strip().upper()

    # Core fields
    st.markdown("#### 📋 Document Summary")
    core_fields = [
        ("vendor_name", "🏢 Vendor"), ("document_number", "🔢 Document #"),
        ("invoice_number", "🔢 Invoice #"), ("document_date", "📅 Date"),
        ("invoice_date", "📅 Invoice Date"), ("due_date", "📅 Due Date"),
        ("currency", "💱 Currency"), ("total_amount", "💰 Total Amount"),
        ("grand_total", "💰 Grand Total"), ("bill_to", "📍 Bill To"),
        ("account_number", "🔑 Account #"), ("customer_name", "👤 Customer"),
        ("account_holder", "👤 Account Holder"), ("bank_name", "🏦 Bank"),
        ("statement_date", "📅 Statement Date"),
        ("billing_period_from", "📅 Period From"), ("billing_period_to", "📅 Period To"),
        ("statement_period_from", "📅 Period From"), ("statement_period_to", "📅 Period To"),
    ]
    core_money_keys = {"total_amount", "grand_total"}
    displayed = []
    for key, label in core_fields:
        if not data.get(key) or str(data.get(key)) == "null":
            continue
        value = data[key]
        if key in core_money_keys:
            value = _format_money_with_currency(value, currency_code)
        displayed.append((label, str(value)))

    if displayed:
        cols = st.columns(min(len(displayed), 3))
        for i, (label, val) in enumerate(displayed):
            with cols[i % 3]:
                display_val = val[:120] + "..." if len(val) > 120 else val
                st.markdown(f"**{label}**")
                st.code(display_val, language=None)

    st.divider()

    # Line items / Transactions
    line_items = data.get("line_items", [])
    transactions = data.get("transactions", [])
    items = line_items or transactions

    if items:
        label = "Transactions" if transactions else "Line Items"
        st.markdown(f"#### 📦 {label} ({len(items)} rows)")
        df = pd.DataFrame(items)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        for money_col in ("Unit Price", "Tax", "Amount"):
            if money_col in df.columns:
                df[money_col] = df[money_col].apply(lambda v: _format_money_with_currency(v, currency_code))
        df = df.dropna(axis=1, how="all")
        if "Low Confidence" in df.columns and not df["Low Confidence"].any():
            df = df.drop(columns=["Low Confidence"])
        st.dataframe(df, use_container_width=True, hide_index=True, height=min(500, 40 + len(df) * 35))

    # Totals
    total_fields = [
        ("subtotal", "Subtotal"), ("tax_total", "Tax Total"), ("discount", "Discount"),
        ("freight_charges", "Freight"), ("grand_total", "Grand Total"),
        ("amount_in_words", "In Words"), ("opening_balance", "Opening Bal"),
        ("closing_balance", "Closing Bal"), ("total_debits", "Total Debits"),
        ("total_credits", "Total Credits"), ("previous_balance", "Prev Balance"),
        ("current_charges", "Current Charges"), ("payment_received", "Payment Recv"),
    ]
    total_money_keys = {
        "subtotal", "tax_total", "discount", "freight_charges", "grand_total",
        "opening_balance", "closing_balance", "total_debits", "total_credits",
        "previous_balance", "current_charges", "payment_received",
    }
    totals = []
    for key, label in total_fields:
        if not data.get(key):
            continue
        value = data[key]
        if key in total_money_keys:
            value = _format_money_with_currency(value, currency_code)
        totals.append((label, value))
    if totals:
        st.markdown("#### 💰 Totals & Summary")
        cols = st.columns(min(len(totals), 4))
        for i, (lbl, val) in enumerate(totals):
            with cols[i % 4]:
                st.metric(lbl, str(val))

    # Surcharges
    surcharges = data.get("surcharges", [])
    if surcharges:
        st.markdown("#### 📎 Surcharges & Levies")
        st.dataframe(pd.DataFrame(surcharges), use_container_width=True, hide_index=True)

    # Additional fields
    addl = data.get("additional_fields", {})
    if addl:
        st.markdown("#### 📎 Additional Fields")
        st.dataframe(pd.DataFrame([{"Field": k, "Value": str(v)} for k, v in addl.items()]), use_container_width=True, hide_index=True)

    # Payment info
    payment = data.get("payment_info", {})
    if payment and any(v for v in payment.values() if v):
        st.markdown("#### 💳 Payment Information")
        for k, v in payment.items():
            if v:
                st.markdown(f"**{k.replace('_', ' ').title()}:** `{v}`")


def display_bank_matching(data: dict, report_path: Path = None):
    """Display bank matching results with visual indicators."""
    bank = data.get("bank_statement_summary", {})

    st.markdown("#### 🏦 Bank Statement Summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Bank", bank.get("bank", "N/A"))
    b2.metric("Account #", str(bank.get("account_no", "N/A")))
    b3.metric("Total Credits", f"MYR {bank.get('total_credits', 0):,.2f}")
    b4.metric("Total Debits", f"MYR {bank.get('total_debits', 0):,.2f}")
    st.markdown(f"**Period:** {bank.get('period', 'N/A')} | **Entries:** {bank.get('total_entries', 0)}")
    st.divider()

    # Documents summary
    docs = data.get("documents_summary", [])
    if docs:
        st.markdown(f"#### 📄 Extracted Documents ({len(docs)} files)")
        df_docs = pd.DataFrame(docs)
        df_docs.columns = [c.replace("_", " ").title() for c in df_docs.columns]
        st.dataframe(df_docs, use_container_width=True, hide_index=True)
        st.divider()

    # Match results
    exact = data.get("exact_matches", [])
    near = data.get("near_matches", [])
    unmatched_bank = data.get("unmatched_bank_entries", [])
    unmatched_docs = data.get("unmatched_documents", [])

    st.markdown("#### 🎯 Matching Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("✅ Exact Matches", len(exact))
    m2.metric("🟡 Near Matches", len(near))
    m3.metric("❌ Unmatched Bank", len(unmatched_bank))
    m4.metric("❌ Unmatched Docs", len(unmatched_docs))

    if exact:
        st.markdown("##### ✅ Exact Matches")
        for m in exact:
            st.success(
                f"**Bank:** {m.get('bank_date','')} | {m.get('bank_description','')} | "
                f"{m.get('bank_type','')} MYR {m.get('bank_amount','')}\n\n"
                f"**Doc:** {m.get('doc_file','')} | #{m.get('doc_number','')} | "
                f"{m.get('doc_vendor','')} | MYR {m.get('doc_amount','')} ({m.get('match_field','')})"
            )

    if near:
        st.markdown("##### 🟡 Near Matches")
        for m in near:
            st.warning(
                f"**Bank:** {m.get('bank_date','')} | {m.get('bank_description','')} | "
                f"MYR {m.get('bank_amount','')}\n\n"
                f"**Doc:** {m.get('doc_file','')} | MYR {m.get('doc_amount','')} | "
                f"Diff: {m.get('difference_pct','?')}%"
            )

    if unmatched_bank:
        st.markdown("##### ❌ Unmatched Bank Entries")
        df_ub = pd.DataFrame(unmatched_bank)
        if not df_ub.empty:
            df_ub.columns = [c.replace("_", " ").title() for c in df_ub.columns]
            st.dataframe(df_ub, use_container_width=True, hide_index=True)

    # Full text report
    if report_path and report_path.exists():
        st.divider()
        with st.expander("📄 View Full Matching Report", expanded=False):
            st.code(report_path.read_text(encoding="utf-8"), language=None)
        st.download_button(
            "⬇️ Download Full Report",
            data=report_path.read_text(encoding="utf-8"),
            file_name="bank_matching_report.txt", mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════
# REPORT FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════════════

REPORT_COLUMNS = [
    "No", "Company Name", "TIN No", "Types (Inv/CN)", "Invoice No",
    "No. Invois Cukai", "Invoice Date", "Tarikh Invois", "No. Akaun",
    "Lot No", "Location", "Account No", "Lease ID", "Unit No", "Project",
    "Premise Address", "Contract No / Batch No", "Contract Account No",
    "Description", "Jumlah perlu dibayar\n(Including tax)", "Amaun Elektrik",
    "LHDN UUID", "Validate On", "Kwh Reading Before", "Kwh Reading After",
    "Current Reading / Total Units",
]


def _safe(d: dict, *keys, default=""):
    """Dig through nested dicts safely."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur if cur not in (None, "") else default


def _format_money_with_currency(value: object, currency: str) -> str:
    if value in (None, ""):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    currency_code = (currency or "").strip().upper()
    if not currency_code:
        return text

    if text.upper().startswith(f"{currency_code} ") or text.upper() == currency_code:
        return text

    return f"{currency_code} {text}"


def _parse_lot_no(data: dict) -> str:
    """Extract Lot No from bill_to or additional_fields."""
    bill_to = _safe(data, "bill_to")
    if isinstance(bill_to, str):
        import re
        m = re.search(r"Lot\s*No\.?\s*[:：]?\s*(\S+)", bill_to, re.IGNORECASE)
        if m:
            return m.group(1)
    return _safe(data, "additional_fields", "Lot No")


def _parse_unit_no(data: dict) -> str:
    """Extract Unit No from line items or additional_fields."""
    af = _safe(data, "additional_fields", "Unit No")
    if af:
        return af
    for item in data.get("line_items", []):
        desc = item.get("description", "") or item.get("product_description", "") or ""
        import re
        m = re.search(r"Unit\s*No\.?\s*[:：]?\s*(\S+)", desc, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def _parse_location(data: dict) -> str:
    """Extract location from line items description or bill_to."""
    for item in data.get("line_items", []):
        desc = item.get("description", "") or ""
        if "Mall" in desc or "Hotel" in desc or "Plaza" in desc:
            import re
            m = re.search(r"(?:The\s+\w+\s+Mall|[\w\s]+Mall|[\w\s]+Hotel|[\w\s]+Plaza)", desc)
            if m:
                return m.group(0).strip()
    return ""


def _parse_kwh_readings(data: dict) -> tuple:
    """Parse kWh meter readings from utility bill line items."""
    readings_before, readings_after, total_units = [], [], []
    for item in data.get("line_items", []):
        desc = item.get("description", "") or ""
        import re
        m = re.search(r"Meter Readings?\s*([\d,]+(?:\.\d+)?)\s*[-–]\s*([\d,]+(?:\.\d+)?)", desc)
        if m:
            before = m.group(1).replace(",", "")
            after = m.group(2).replace(",", "")
            if before not in readings_before:
                readings_before.append(before)
            if after not in readings_after:
                readings_after.append(after)
        qty = item.get("quantity")
        if qty and desc and "Meter" in desc:
            total_units.append(str(qty))
    return (
        " / ".join(readings_before) if readings_before else "",
        " / ".join(readings_after) if readings_after else "",
        " / ".join(total_units) if total_units else "",
    )


def _electricity_amount(data: dict) -> str:
    """Sum electricity charge amounts for utility bills."""
    for item in data.get("line_items", []):
        desc = (item.get("description", "") or "").lower()
        if "electricity" in desc and item.get("amount"):
            return str(item["amount"])
    return ""


def _build_description(data: dict) -> str:
    """Build a short description from line items."""
    descs = []
    for item in data.get("line_items", []):
        d = item.get("description") or item.get("product_description") or ""
        first_line = d.split("\n")[0].strip()
        if first_line and first_line not in descs:
            descs.append(first_line)
    return "; ".join(descs[:4]) + ("..." if len(descs) > 4 else "")


def _doc_type_label(data: dict) -> str:
    """Return the document type label (Inv / CN / Utility / etc.)."""
    dt = (_safe(data, "document_type") or "").lower()
    if "credit" in dt:
        return "CN"
    if "utility" in dt:
        return "Utility"
    if "rental" in dt or "lease" in dt:
        return "Rental"
    if "statement" in dt:
        return "SOA"
    if "hotel" in dt:
        return "Hotel"
    if "travel" in dt:
        return "Travel"
    return "Inv"


def map_extraction_to_report_row(data: dict, index: int) -> dict:
    """Map a single extraction JSON to a report format row."""
    af = data.get("additional_fields", {}) or {}
    currency_code = str(_safe(data, "currency") or "").strip().upper()
    kwh_before, kwh_after, total_units = _parse_kwh_readings(data)

    invoice_no = (
        _safe(data, "invoice_number")
        or _safe(data, "document_number")
        or _safe(data, "statement_number")
        or ""
    )
    invoice_date = (
        _safe(data, "invoice_date")
        or _safe(data, "document_date")
        or _safe(data, "statement_date")
        or _safe(af, "Invoice Date")
        or ""
    )
    account_no = (
        _safe(data, "account_number")
        or _safe(data, "customer_account")
        or _safe(af, "Account No")
        or _safe(data, "payment_info", "account_number")
        or ""
    )

    return {
        "No": index,
        "Company Name": _safe(data, "vendor_name"),
        "TIN No": _safe(af, "TIN No.") or _safe(af, "TIN No"),
        "Types (Inv/CN)": _doc_type_label(data),
        "Invoice No": invoice_no,
        "No. Invois Cukai": _safe(af, "No. Invois Cukai") or _safe(af, "Tax Invoice No"),
        "Invoice Date": invoice_date,
        "Tarikh Invois": invoice_date,
        "No. Akaun": account_no,
        "Lot No": _parse_lot_no(data),
        "Location": _parse_location(data),
        "Account No": account_no,
        "Lease ID": _safe(af, "Lease ID"),
        "Unit No": _parse_unit_no(data),
        "Project": _safe(af, "Project") or _safe(af, "Project Name"),
        "Premise Address": _safe(data, "service_address") or _safe(data, "bill_to"),
        "Contract No / Batch No": _safe(af, "Contract No") or _safe(af, "Batch No") or _safe(af, "Contract No / Batch No"),
        "Contract Account No": _safe(af, "Contract Account No"),
        "Description": _build_description(data),
        "Jumlah perlu dibayar\n(Including tax)": _format_money_with_currency(
            _safe(data, "grand_total") or _safe(data, "total_amount"),
            currency_code,
        ),
        "Amaun Elektrik": _format_money_with_currency(_electricity_amount(data), currency_code),
        "LHDN UUID": _safe(af, "LHDN UUID") or _safe(af, "e-Invoice UUID"),
        "Validate On": _safe(af, "Validate On") or _safe(af, "Validated On"),
        "Kwh Reading Before": kwh_before,
        "Kwh Reading After": kwh_after,
        "Current Reading / Total Units": total_units,
    }


def load_all_extraction_rows() -> pd.DataFrame:
    """Load all extraction JSON files and map to report DataFrame."""
    extraction_dir = Path(__file__).resolve().parent / "extraction_output"
    rows = []
    if extraction_dir.exists():
        files = sorted(
            [
                f for f in extraction_dir.glob("*.json")
                if f.name not in {"bank_matching_results.json"}
            ]
        )
        for i, f in enumerate(files, start=1):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    row = map_extraction_to_report_row(data, i)
                    row["_source_file"] = f.name
                    rows.append(row)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    df = pd.DataFrame(rows)
    # Ensure all report columns exist
    for col in REPORT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _team_from_doc_type(doc_type: str) -> str:
    text = (doc_type or "").strip().lower()
    if "rental" in text or "lease" in text or "utility" in text:
        return "rental"
    return "sales"


def _doc_team_map_path() -> Path:
    return Path(__file__).resolve().parent / "docs" / "database" / "doc_teams.json"


def load_doc_team_map() -> dict[str, str]:
    path = _doc_team_map_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v).lower() for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_doc_team_map(doc_team_map: dict[str, str]) -> None:
    path = _doc_team_map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc_team_map, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_document_team(file_path: Path, doc_team_map: dict[str, str]) -> str:
    mapped = doc_team_map.get(file_path.name)
    if mapped in {"sales", "rental"}:
        return mapped

    extraction_dir = Path(__file__).resolve().parent / "extraction_output"
    if extraction_dir.exists():
        candidates = sorted(
            [
                p for p in extraction_dir.glob("*.json")
                if p.name not in {"bank_matching_results.json"}
                and (p.stem == file_path.stem or p.stem.startswith(file_path.stem + "_extracted"))
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return _team_from_doc_type(str(data.get("document_type") or ""))
            except Exception:
                continue

    name_l = file_path.name.lower()
    if any(token in name_l for token in ["rental", "lease", "ll_"]):
        return "rental"
    return "sales"


def find_source_pdf_for_extraction(source_file: str) -> Path | None:
    """Match extraction JSON source filename to original PDF in docs/database."""
    if not source_file:
        return None

    database_dir = Path(__file__).resolve().parent / "docs" / "database"
    if not database_dir.exists():
        return None

    source_stem = Path(source_file).stem
    base = source_stem.replace("_extracted", "")

    exact_name = f"{base}.pdf"
    exact_path = database_dir / exact_name
    if exact_path.exists():
        return exact_path

    pdf_files = [p for p in database_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    for pdf in pdf_files:
        if pdf.name.lower() == exact_name.lower():
            return pdf

    for pdf in pdf_files:
        if pdf.stem.lower().startswith(base.lower()):
            return pdf

    return None


def load_extraction_repository_items() -> list[dict]:
    """Build row data for Extraction Viewer repository layout."""
    extraction_dir = Path(__file__).resolve().parent / "extraction_output"
    if not extraction_dir.exists():
        return []

    items: list[dict] = []
    files = sorted(
        [
            f for f in extraction_dir.glob("*.json")
            if f.name not in {"bank_matching_results.json"}
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        additional = data.get("additional_fields", {}) or {}
        invoice_id = (
            data.get("invoice_number")
            or data.get("document_number")
            or data.get("statement_number")
            or f.stem.replace("_extracted", "")
        )
        vendor = (
            data.get("vendor_name")
            or data.get("customer_name")
            or data.get("account_holder")
            or "-"
        )
        date_text = (
            data.get("invoice_date")
            or data.get("document_date")
            or data.get("statement_date")
            or "-"
        )
        currency_code = str(data.get("currency") or "").strip().upper()
        total_raw = data.get("grand_total") or data.get("total_amount") or data.get("subtotal") or ""
        total_text = _format_money_with_currency(total_raw, currency_code) if total_raw else "-"
        team = _team_from_doc_type(str(data.get("document_type") or ""))

        status_text = st.session_state["processing_doc_status"].get(f.name, "Ready for Review")

        items.append(
            {
                "invoice_id": str(invoice_id),
                "vendor": str(vendor),
                "date": str(date_text),
                "total": str(total_text),
                "status": str(status_text),
                "team": team,
                "last_updated": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d"),
                "source_file": f.name,
                "data": data,
            }
        )

    return items


def _normalize_lease_id(lid: str) -> str:
    """Strip non-digit prefixes from Lease ID for fuzzy comparison."""
    import re
    if not lid:
        return ""
    return re.sub(r"[^\d]", "", str(lid))


def match_utility_to_rental(df: pd.DataFrame) -> list[dict]:
    """
    Match utility bills to rental invoices using shared identifiers.
    Returns a list of match dicts:  {utility_idx, rental_idx, matched_on, keys}
    Priority: Lease ID > Lot No > (Vendor + TIN No)
    """
    utilities = df[df["Types (Inv/CN)"] == "Utility"]
    rentals = df[df["Types (Inv/CN)"].isin(["Rental", "Inv"])]
    matches = []
    matched_rental_ids = set()

    for u_idx, u_row in utilities.iterrows():
        best_match = None
        best_score = 0
        best_keys = []

        u_lease = _normalize_lease_id(u_row.get("Lease ID", ""))
        u_lot = str(u_row.get("Lot No", "")).strip()
        u_vendor = str(u_row.get("Company Name", "")).strip().lower()
        u_tin = str(u_row.get("TIN No", "")).strip()
        u_acct = str(u_row.get("Account No", "")).strip()

        for r_idx, r_row in rentals.iterrows():
            if r_idx in matched_rental_ids:
                continue
            score = 0
            keys = []

            r_lease = _normalize_lease_id(r_row.get("Lease ID", ""))
            if u_lease and r_lease and u_lease == r_lease:
                score += 4
                keys.append(f"Lease ID ({u_row.get('Lease ID', '')} ↔ {r_row.get('Lease ID', '')})")

            r_lot = str(r_row.get("Lot No", "")).strip()
            if u_lot and r_lot and u_lot == r_lot:
                score += 3
                keys.append(f"Lot No ({u_lot})")

            r_vendor = str(r_row.get("Company Name", "")).strip().lower()
            if u_vendor and r_vendor and u_vendor == r_vendor:
                score += 2
                keys.append("Vendor Name")

            r_tin = str(r_row.get("TIN No", "")).strip()
            if u_tin and r_tin and u_tin == r_tin:
                score += 2
                keys.append(f"TIN No ({u_tin})")

            r_acct = str(r_row.get("Account No", "")).strip()
            if u_acct and r_acct and u_acct == r_acct:
                score += 1
                keys.append(f"Account No ({u_acct})")

            if score > best_score:
                best_score = score
                best_match = r_idx
                best_keys = keys

        if best_match is not None and best_score >= 2:
            matched_rental_ids.add(best_match)
            matches.append({
                "utility_idx": u_idx,
                "rental_idx": best_match,
                "score": best_score,
                "matched_on": best_keys,
            })

    return matches


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>📄 Watsons Document Intelligence Platform</h1>
    <p>AI-Powered OCR &bull; Document Classification &bull; Data Extraction &bull; Bank Statement Matching</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Logo / Brand ──────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-logo">'
        '<span class="sidebar-logo-icon">🔬</span>'
        '<div class="sidebar-logo-text">Watsons Intelligence</div>'
        '<div class="sidebar-logo-sub">Document AI Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.markdown('<div class="sidebar-section">Access</div>', unsafe_allow_html=True)
    role_label = st.selectbox("Current Role", ["Admin", "Sales", "Rental"], key="current_role_label")
    current_role = role_label.lower()

    st.markdown("")
    st.divider()

    # ── Navigation ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    _nav_pages = [
        "📤 Document Processing",
        "🔍 OCR Viewer",
        "📊 Extraction Viewer",
        "📋 Report Format",
    ]
    if current_role == "admin":
        _nav_pages.append("🏦 Bank Matching")
    page = st.radio(
        "nav",
        _nav_pages,
        label_visibility="collapsed",
    )

    st.markdown("")
    st.divider()

    # ── Capabilities Card ─────────────────────────────────────
    st.markdown('<div class="sidebar-section">Capabilities</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-about-card">'
        '<div class="sidebar-about-item"><span>📄</span><span><strong>OCR</strong> — extract text from scanned docs</span></div>'
        '<div class="sidebar-about-item"><span>🏷️</span><span><strong>Classify</strong> — identify document types</span></div>'
        '<div class="sidebar-about-item"><span>📊</span><span><strong>Extract</strong> — pull structured financial data</span></div>'
        '<div class="sidebar-about-item"><span>📋</span><span><strong>Report</strong> — generate formatted reports</span></div>'
        '<div class="sidebar-about-item"><span>🏦</span><span><strong>Match</strong> — reconcile against bank statements</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.divider()

    # ── Footer ─────────────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-footer">'
        f'<span class="sidebar-badge">v1.0</span><br/>'
        f'<span style="color:#70c9d2;font-size:0.7rem;">Powered by Azure OpenAI &bull; {datetime.now().strftime("%d %b %Y")}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════
for key in ("ocr_result", "extraction_result", "doc_type", "uploaded_images", "processing_stage"):
    if key not in st.session_state:
        st.session_state[key] = None
if "doc_status" not in st.session_state:
    st.session_state["doc_status"] = {}  # {row_no: "verified"|"rejected"|"pending"}
if "processing_doc_status" not in st.session_state:
    st.session_state["processing_doc_status"] = {}
if "processing_selected_doc" not in st.session_state:
    st.session_state["processing_selected_doc"] = None
if "extraction_selected_file" not in st.session_state:
    st.session_state["extraction_selected_file"] = None
if "report_preview_source" not in st.session_state:
    st.session_state["report_preview_source"] = None
if "report_detail_row" not in st.session_state:
    st.session_state["report_detail_row"] = None


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
if page == "📤 Document Processing":

    st.markdown("### 📤 Document Processing Pipeline")
    st.markdown("Upload a PDF to run the full AI pipeline: **PDF → Images → OCR → Classification → Extraction**")

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"],
            help="Invoices, Utility Bills, Bank Statements, Travel, Rental, SOA, etc.")
    with col_opt:
        force_type = st.selectbox("Force document type (optional)", ["Auto-detect"] + list(AGENT_REGISTRY.keys()))
        ocr_mode = st.radio("OCR Mode", ["Batch (all pages)", "Per-page"], index=0)
        upload_team_choice = st.selectbox("Document Team", ["Auto", "Sales", "Rental"], index=0)

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        app_dir = Path(__file__).resolve().parent
        database_dir = app_dir / "docs" / "database"
        database_dir.mkdir(parents=True, exist_ok=True)

        database_path = database_dir / uploaded_file.name
        if database_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            database_path = database_dir / f"{database_path.stem}_{stamp}{database_path.suffix}"
        database_path.write_bytes(uploaded_bytes)

        doc_team_map = load_doc_team_map()
        assigned_team = upload_team_choice.lower()
        if assigned_team == "auto":
            assigned_team = _team_from_doc_type(force_type if force_type != "Auto-detect" else "")
        doc_team_map[database_path.name] = assigned_team
        save_doc_team_map(doc_team_map)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / uploaded_file.name
            pdf_path.write_bytes(uploaded_bytes)
            st.markdown(f"**📄 Uploaded:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
            st.caption(f"Stored in database: `{database_path.name}` | Team: `{assigned_team.title()}`")

            if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
                with st.status("🔄 Processing document...", expanded=True) as status:
                    progress = st.progress(0)

                    # Step 1: PDF to Images
                    st.write("**Step 1/4:** Converting PDF to images...")
                    try:
                        image_dir = Path(tmp_dir) / f"{pdf_path.stem}_images"
                        image_paths = pdf_to_images(pdf_path, image_dir)
                        progress.progress(20)
                        st.write(f"  ✅ Converted to **{len(image_paths)} page(s)**")
                        tcols = st.columns(min(len(image_paths), 6))
                        for i, ip in enumerate(image_paths[:6]):
                            with tcols[i]:
                                st.image(str(ip), caption=f"Page {i+1}", width=110)
                    except Exception as e:
                        st.error(f"❌ PDF conversion failed: {e}")
                        st.stop()

                    # Step 2: OCR
                    st.write("**Step 2/4:** Running AI-powered OCR...")
                    try:
                        user_prompt = (
                            "Transcribe ALL visible text from this document image exactly as it appears. "
                            "Output the result as a single valid JSON object following the schema in your instructions. "
                            "Do NOT interpret, summarize, or calculate anything. "
                            "Preserve all numbers, punctuation, and formatting exactly."
                        )
                        if ocr_mode.startswith("Batch"):
                            raw_ocr = ocr_images_with_chat_model(image_paths, user_prompt)
                            ocr_parsed = _maybe_parse_json(raw_ocr)
                            ocr_json_str = raw_ocr if isinstance(raw_ocr, str) else json.dumps(ocr_parsed, ensure_ascii=False)
                        else:
                            pages_list = []
                            for idx, ip in enumerate(image_paths):
                                st.write(f"  OCR page {idx+1}/{len(image_paths)}...")
                                raw = ocr_image_with_chat_model(ip, user_prompt)
                                pages_list.append({"page_number": idx+1, "file": ip.name, "model_output": _maybe_parse_json(raw)})
                            ocr_parsed = {"mode": "per_image", "results": pages_list}
                            ocr_json_str = json.dumps(ocr_parsed, ensure_ascii=False)
                        st.session_state.ocr_result = ocr_parsed
                        progress.progress(60)
                        st.write("  ✅ OCR complete")
                    except Exception as e:
                        st.error(f"❌ OCR failed: {e}")
                        st.stop()

                    # Step 3 & 4: Classify + Extract
                    st.write("**Step 3/4:** Classifying & extracting...")
                    try:
                        forced = None if force_type == "Auto-detect" else force_type
                        doc_type_result, extracted = orchestrator_run(ocr_json_str, forced_type=forced)
                        st.session_state.doc_type = doc_type_result
                        st.session_state.extraction_result = extracted
                        progress.progress(95)
                        st.write(f"  ✅ Classified as: **{doc_type_result.replace('_',' ').title()}**")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {e}")
                        st.stop()

                    progress.progress(100)
                    status.update(label="✅ Pipeline complete!", state="complete", expanded=True)

                st.divider()
                st.markdown("### 📋 Results")
                tab_ext, tab_ocr, tab_json = st.tabs(["📊 Extracted Data", "🔍 OCR Output", "📝 Raw JSON"])
                with tab_ext:
                    display_extraction_result(extracted, doc_type_result)
                with tab_ocr:
                    display_ocr_result(ocr_parsed if isinstance(ocr_parsed, dict) else {"raw": ocr_parsed})
                with tab_json:
                    jc1, jc2 = st.columns(2)
                    with jc1:
                        st.markdown("**OCR Output**")
                        st.json(ocr_parsed)
                    with jc2:
                        st.markdown("**Extraction Output**")
                        st.json(extracted)

                save_base_name = pdf_path.stem
                app_dir = Path(__file__).resolve().parent
                ocr_output_dir = app_dir / "ocr_output"
                extraction_output_dir = app_dir / "extraction_output"
                ocr_output_dir.mkdir(parents=True, exist_ok=True)
                extraction_output_dir.mkdir(parents=True, exist_ok=True)

                ocr_output_path = ocr_output_dir / f"{save_base_name}.json"
                extraction_output_path = extraction_output_dir / f"{save_base_name}.json"

                with open(ocr_output_path, "w", encoding="utf-8") as f:
                    json.dump(ocr_parsed, f, ensure_ascii=False, indent=2)
                with open(extraction_output_path, "w", encoding="utf-8") as f:
                    json.dump(extracted, f, ensure_ascii=False, indent=2)

                st.success(
                    f"Saved results to `{ocr_output_path.name}` and `{extraction_output_path.name}`. "
                    "These are now available in OCR Viewer, Extraction Viewer, and Report Format."
                )

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button("⬇️ Download OCR JSON",
                        data=json.dumps(ocr_parsed, ensure_ascii=False, indent=2),
                        file_name=f"{pdf_path.stem}_ocr.json", mime="application/json")
                with dc2:
                    st.download_button("⬇️ Download Extraction JSON",
                        data=json.dumps(extracted, ensure_ascii=False, indent=2),
                        file_name=f"{pdf_path.stem}_extracted.json", mime="application/json")

    st.divider()

    st.markdown("### Documents")
    st.caption("Upload and manage document processing")

    database_dir = Path(__file__).resolve().parent / "docs" / "database"
    source_docs = (
        sorted([p for p in database_dir.iterdir() if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
        if database_dir.exists()
        else []
    )
    doc_team_map = load_doc_team_map()
    source_docs_with_team = [(p, infer_document_team(p, doc_team_map)) for p in source_docs]
    visible_source_docs = (
        source_docs_with_team
        if current_role == "admin"
        else [(p, team) for p, team in source_docs_with_team if team == current_role]
    )

    st.markdown(f"#### All Documents ({len(visible_source_docs)})")

    if visible_source_docs:
        h1, h2, h3, h4, h5, h6, h7 = st.columns([1.2, 3.2, 1.0, 1.0, 1.3, 1.7, 2.4])
        h1.markdown("**Doc ID**")
        h2.markdown("**File Name**")
        h3.markdown("**Type**")
        h4.markdown("**Size (MB)**")
        h5.markdown("**Upload Date**")
        h6.markdown("**Status**")
        h7.markdown("**Actions**")

        for idx, (file_path, _team) in enumerate(visible_source_docs, start=1):
            display_file_name = file_path.name
            file_type = file_path.suffix.replace(".", "").upper() or "FILE"
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            upload_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")
            status = st.session_state["processing_doc_status"].get(file_path.name, "Ready for Review")

            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 3.2, 1.0, 1.0, 1.3, 1.7, 2.4])
            c1.markdown(f"**DOC-{idx:04d}**")
            c2.markdown(display_file_name)
            c3.markdown(file_type)
            c4.markdown(f"{file_size_mb:.1f}")
            c5.markdown(upload_date)
            c6.markdown(status)

            is_viewing = st.session_state.get("processing_selected_doc") == str(file_path)
            btn_label = "Hide" if is_viewing else "View"
            if c7.button(btn_label, key=f"view_doc_{file_path.name}", use_container_width=True):
                if is_viewing:
                    st.session_state["processing_selected_doc"] = None
                else:
                    st.session_state["processing_selected_doc"] = str(file_path)
                st.rerun()

            if st.session_state.get("processing_selected_doc") == str(file_path):
                st.markdown(f"##### Preview: {file_path.name}")
                display_processing_file_preview(file_path)

            st.markdown("---")

    else:
        st.info(f"No documents found for role: {current_role.title()}.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OCR VIEWER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 OCR Viewer":

    st.markdown("### 🔍 OCR Output Viewer")
    st.markdown("Browse previously processed OCR results with confidence scoring.")

    ocr_dir = Path(__file__).resolve().parent / "ocr_output"
    if ocr_dir.exists():
        ocr_files = sorted(ocr_dir.glob("*.json"))
        if ocr_files:
            sel = st.selectbox("Select OCR output", ocr_files, format_func=lambda p: p.name)
            if sel:
                data = load_json_file(sel)
                if isinstance(data, dict):
                    display_ocr_result(data)
                else:
                    st.json(data)
        else:
            st.info("No OCR output files found.")
    else:
        st.info("OCR output directory not found.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: EXTRACTION VIEWER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Extraction Viewer":

    st.markdown("### 📊 Extraction Viewer")
    st.markdown("Invoice repository and details")
    st.caption(f"Current access role: {current_role.title()}")

    repo_items = load_extraction_repository_items()
    if not repo_items:
        st.info("No extraction files found.")
    else:
        search_query = st.text_input("", placeholder="Search invoice ID, vendor, or source file...", label_visibility="collapsed")
        f1, f2 = st.columns([1, 1])
        with f1:
            all_statuses = sorted({item["status"] for item in repo_items if item["status"]})
            status_filter = st.selectbox("Status", ["All Statuses"] + all_statuses)
        with f2:
            all_vendors = sorted({item["vendor"] for item in repo_items if item["vendor"] and item["vendor"] != "-"})
            vendor_filter = st.selectbox("Vendor", ["All Vendors"] + all_vendors)

        filtered_items = repo_items if current_role == "admin" else [item for item in repo_items if item.get("team") == current_role]

        selected_file = st.session_state.get("extraction_selected_file")
        if selected_file and all(item["source_file"] != selected_file for item in filtered_items):
            st.session_state["extraction_selected_file"] = None

        if search_query:
            q = search_query.lower().strip()
            filtered_items = [
                item for item in filtered_items
                if q in item["invoice_id"].lower()
                or q in item["vendor"].lower()
                or q in item["source_file"].lower()
            ]
        if status_filter != "All Statuses":
            filtered_items = [item for item in filtered_items if item["status"] == status_filter]
        if vendor_filter != "All Vendors":
            filtered_items = [item for item in filtered_items if item["vendor"] == vendor_filter]

        st.markdown(f"#### Invoices ({len(filtered_items)})")
        h1, h2, h3, h4, h5, h6 = st.columns([2.3, 3.0, 1.5, 1.7, 2.0, 1.5])
        h1.markdown("**Invoice ID**")
        h2.markdown("**Vendor**")
        h3.markdown("**Date**")
        h4.markdown("**Total**")
        h5.markdown("**Status**")
        h6.markdown("**Last Updated**")

        for item in filtered_items:
            r1, r2, r3, r4, r5, r6 = st.columns([2.3, 3.0, 1.5, 1.7, 2.0, 1.5])
            if r1.button(item["invoice_id"], key=f"ext_row_{item['source_file']}", use_container_width=True):
                current = st.session_state.get("extraction_selected_file")
                st.session_state["extraction_selected_file"] = None if current == item["source_file"] else item["source_file"]

            r2.markdown(item["vendor"])
            r3.markdown(item["date"])
            r4.markdown(item["total"])
            r5.markdown(item["status"])
            r6.markdown(item["last_updated"])

            if st.session_state.get("extraction_selected_file") == item["source_file"]:
                st.markdown(f"##### Details: {item['invoice_id']} ({item['source_file']})")
                detail_tab, raw_tab = st.tabs(["📊 Structured View", "📝 Raw JSON"])
                with detail_tab:
                    display_extraction_result(item["data"], item["data"].get("document_type", "Unknown"))
                with raw_tab:
                    st.json(item["data"])

            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: REPORT FORMAT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Report Format":

    st.markdown("### 📋 Extraction Report — Spreadsheet View")
    st.markdown(
        "All extracted documents are mapped to the standard report format below. "
        "You can **review, edit, and export** the data."
    )

    df = load_all_extraction_rows()

    if not df.empty and current_role != "admin":
        if current_role == "rental":
            df = df[
                df["Types (Inv/CN)"]
                .astype(str)
                .str.lower()
                .str.contains("rental|lease|utility", regex=True)
            ].copy()
        elif current_role == "sales":
            df = df[
                ~df["Types (Inv/CN)"]
                .astype(str)
                .str.lower()
                .str.contains("rental|lease|utility", regex=True)
            ].copy()

    if df.empty:
        st.info(f"No extraction files available for role: {current_role.title()}.")
    else:
        # ── Summary metrics ──────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total Documents", len(df))
        types_counts = df["Types (Inv/CN)"].value_counts()
        col2.metric("🧾 Invoices", types_counts.get("Inv", 0))
        col3.metric("⚡ Utility Bills", types_counts.get("Utility", 0))
        col4.metric("🏠 Rental", types_counts.get("Rental", 0))

        st.divider()

        # ── Run matching & enrich DataFrame ──────────────────────────
        matches = match_utility_to_rental(df)
        df["Matched To"] = ""
        df["Match Confidence"] = ""
        df["Matched On"] = ""
        for m in matches:
            u_idx, r_idx = m["utility_idx"], m["rental_idx"]
            conf = "High" if m["score"] >= 6 else ("Medium" if m["score"] >= 4 else "Low")
            keys_str = ", ".join(m["matched_on"])
            # Utility row → points to rental
            r_inv = df.loc[r_idx, "Invoice No"] if r_idx in df.index else ""
            r_name = df.loc[r_idx, "Company Name"] if r_idx in df.index else ""
            df.at[u_idx, "Matched To"] = f"Rental #{df.loc[r_idx, 'No']} — {r_inv} ({r_name})"
            df.at[u_idx, "Match Confidence"] = conf
            df.at[u_idx, "Matched On"] = keys_str
            # Rental row → points to utility
            u_inv = df.loc[u_idx, "Invoice No"] if u_idx in df.index else ""
            u_name = df.loc[u_idx, "Company Name"] if u_idx in df.index else ""
            df.at[r_idx, "Matched To"] = f"Utility #{df.loc[u_idx, 'No']} — {u_inv} ({u_name})"
            df.at[r_idx, "Match Confidence"] = conf
            df.at[r_idx, "Matched On"] = keys_str
            # Copy utility-specific fields into the rental row
            for fld in ["Amaun Elektrik", "Kwh Reading Before", "Kwh Reading After", "Current Reading / Total Units"]:
                u_val = df.loc[u_idx, fld] if u_idx in df.index else ""
                if u_val and (not df.at[r_idx, fld]):
                    df.at[r_idx, fld] = u_val

        # ── Tabs: Spreadsheet  |  Matched Pairs ──────────────────────
        tab_sheet, tab_matched = st.tabs(["📊 Spreadsheet View", "🔗 Utility ↔ Rental Matching"])

        # ── TAB 1: DOCUMENT REVIEW ────────────────────────────────
        with tab_sheet:
            # Type style lookup
            _type_style = {
                "Inv": ("🧾", "#e0f7f9", "#006770"),
                "Utility": ("⚡", "#f0fafb", "#00856e"),
                "Rental": ("🏠", "#e0f7f0", "#004d55"),
                "Hotel": ("🏨", "#fde8e7", "#EE2D25"),
                "Travel": ("✈️", "#e0f7f9", "#00A0AF"),
                "SOA": ("📑", "#f0fafb", "#007a85"),
                "CN": ("📌", "#fde8e7", "#d68000"),
            }

            # ── Toolbar ──────────────────────────────────────────────
            tb1, tb2, tb3, tb4 = st.columns([2, 2, 2, 1])
            with tb1:
                type_filter = st.multiselect(
                    "Document Type",
                    options=sorted(df["Types (Inv/CN)"].unique()),
                    default=sorted(df["Types (Inv/CN)"].unique()),
                )
            with tb2:
                company_filter = st.multiselect(
                    "Company Name",
                    options=sorted(df["Company Name"].unique()),
                    default=sorted(df["Company Name"].unique()),
                )
            with tb3:
                status_filter = st.multiselect(
                    "Review Status",
                    options=["Pending", "Verified", "Rejected"],
                    default=["Pending", "Verified", "Rejected"],
                )
            with tb4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Reset All", use_container_width=True):
                    st.session_state["doc_status"] = {}
                    st.rerun()

            # Apply filters
            filtered = df[
                (df["Types (Inv/CN)"].isin(type_filter)) & (df["Company Name"].isin(company_filter))
            ].copy()
            # Apply status filter
            status_map_lower = {"Pending": "pending", "Verified": "verified", "Rejected": "rejected"}
            allowed_statuses = {status_map_lower[s] for s in status_filter}
            filtered = filtered[
                filtered["No"].apply(
                    lambda n: st.session_state["doc_status"].get(int(n), "pending") in allowed_statuses
                )
            ]

            # ── Counts bar ───────────────────────────────────────────
            n_verified = sum(1 for v in st.session_state["doc_status"].values() if v == "verified")
            n_rejected = sum(1 for v in st.session_state["doc_status"].values() if v == "rejected")
            n_pending = len(df) - n_verified - n_rejected
            cb1, cb2, cb3, cb4 = st.columns(4)
            cb1.markdown(f"**{len(filtered)}** of **{len(df)}** shown")
            cb2.markdown(f"✅ **{n_verified}** verified")
            cb3.markdown(f"❌ **{n_rejected}** rejected")
            cb4.markdown(f"⏳ **{n_pending}** pending")
            st.markdown("---")

            # ── Document cards ───────────────────────────────────────
            for _, row in filtered.iterrows():
                row_no = int(row["No"])
                doc_type = str(row.get("Types (Inv/CN)", "Inv"))
                icon, type_bg, accent = _type_style.get(doc_type, ("📄", "#f7fafc", "#4a5568"))
                company = row.get("Company Name", "") or "—"
                inv_no = row.get("Invoice No", "") or "—"
                inv_date = row.get("Invoice Date", "") or "—"
                total = row.get("Jumlah perlu dibayar\n(Including tax)", "") or "—"
                matched_to = row.get("Matched To", "")
                status = st.session_state["doc_status"].get(row_no, "pending")
                status_class = f"doc-status-{status}"

                # ── Card header (HTML) ────────────────────────────────
                match_html = ""
                if matched_to:
                    conf = row.get("Match Confidence", "")
                    conf_colors = {"High": "#00856e", "Medium": "#d68000", "Low": "#EE2D25"}
                    mc = conf_colors.get(conf, "#5a8a8f")
                    match_html = f'<span class="doc-card-match" style="background:{mc};color:white;">🔗 {conf}</span>'

                status_icons = {"verified": "✅", "rejected": "❌", "pending": "⏳"}
                status_labels = {"verified": "Verified", "rejected": "Rejected", "pending": "Pending"}
                s_icon = status_icons.get(status, "")
                s_label = status_labels.get(status, "")

                card_html = (
                    f'<div class="doc-card {status_class}">'
                    f'<div class="doc-card-header">'
                    f'<span class="doc-card-num">{row_no}</span>'
                    f'<span class="doc-card-type" style="background:{type_bg};color:{accent};">{icon} {doc_type}</span>'
                    f'<span class="doc-card-company">{company}</span>'
                    f'<span class="doc-card-detail">{inv_no}</span>'
                    f'<span class="doc-card-detail">{inv_date}</span>'
                    f'<span class="doc-card-amount">{total}</span>'
                    f'{match_html}'
                    f'<span class="doc-card-status">{s_icon} {s_label}</span>'
                    f'</div></div>'
                )
                row_left, row_right = st.columns([12, 2])
                with row_left:
                    st.markdown(card_html, unsafe_allow_html=True)
                with row_right:
                    if st.button("View", key=f"view_detail_{row_no}", use_container_width=True):
                        current_open = st.session_state.get("report_detail_row")
                        st.session_state["report_detail_row"] = None if current_open == row_no else row_no

                if st.session_state.get("report_detail_row") == row_no:
                    # Action buttons row
                    ac1, ac2, ac3 = st.columns([1, 1, 4])
                    with ac1:
                        if st.button("✅ Verify", key=f"verify_{row_no}", use_container_width=True,
                                     type="primary" if status != "verified" else "secondary"):
                            st.session_state["doc_status"][row_no] = "verified"
                            st.rerun()
                    with ac2:
                        if st.button("❌ Reject", key=f"reject_{row_no}", use_container_width=True,
                                     type="primary" if status != "rejected" else "secondary"):
                            st.session_state["doc_status"][row_no] = "rejected"
                            st.rerun()
                    with ac3:
                        if status != "pending":
                            if st.button("↩️ Reset to Pending", key=f"reset_{row_no}"):
                                st.session_state["doc_status"][row_no] = "pending"
                                st.rerun()

                    st.markdown("**Quick Reference (table order)**")
                    ordered_cols = [c for c in REPORT_COLUMNS if c in row.index]
                    detail_row = {c: (row.get(c, "") if row.get(c, "") not in (None, "") else "—") for c in ordered_cols}
                    st.dataframe(
                        pd.DataFrame([detail_row]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if matched_to:
                        st.caption(
                            f"Matched To: {matched_to} | Confidence: {row.get('Match Confidence', '')} | Matched On: {row.get('Matched On', '')}"
                        )

                    src = row.get("_source_file", "")
                    if src:
                        st.caption(f"Source: {src}")

                    if src:
                        pdf_match = find_source_pdf_for_extraction(src)
                        btn_label = "📄 View Original PDF"
                        if st.button(btn_label, key=f"view_original_pdf_{row_no}"):
                            current_preview = st.session_state.get("report_preview_source")
                            st.session_state["report_preview_source"] = None if current_preview == src else src

                        if st.session_state.get("report_preview_source") == src:
                            if pdf_match and pdf_match.exists():
                                st.markdown(f"##### Original PDF: {pdf_match.name}")
                                display_processing_file_preview(pdf_match)
                            else:
                                st.info("Matching PDF not found in src/docs/database.")

            st.markdown("---")

            # ── Table View ────────────────────────────────────────────
            st.markdown("#### 📊 Table View")
            display_cols = [c for c in REPORT_COLUMNS if c in filtered.columns]
            for extra in ["Matched To", "Match Confidence", "Matched On"]:
                if extra in filtered.columns:
                    display_cols.append(extra)
            table_df = filtered[display_cols].copy()
            if "No" in table_df.columns:
                table_df.insert(1, "Status", table_df["No"].apply(
                    lambda n: st.session_state["doc_status"].get(int(n), "pending").capitalize()
                ))
            else:
                table_df.insert(0, "Status", "Pending")

            if table_df.empty:
                st.info("No records match the selected filters.")
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Export ────────────────────────────────────────────────
            # Add status column to export
            display_cols = [c for c in REPORT_COLUMNS if c in filtered.columns]
            for extra in ["Matched To", "Match Confidence", "Matched On"]:
                if extra in filtered.columns:
                    display_cols.append(extra)
            export_df = filtered[display_cols].copy()
            if "No" in export_df.columns:
                export_df.insert(1, "Status", export_df["No"].apply(
                    lambda n: st.session_state["doc_status"].get(int(n), "pending").capitalize()
                ))
            else:
                export_df.insert(0, "Status", "Pending")

            st.markdown("#### 📥 Export Report")
            st.caption("Only verified & pending documents shown. Rejected documents are excluded from export.")

            # Filter out rejected for export
            export_clean = export_df[export_df["Status"] != "Rejected"]

            exp_col1, exp_col2, exp_col3 = st.columns(3)

            with exp_col1:
                csv_data = export_clean.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with exp_col2:
                st.download_button(
                    "⬇️ Download Excel",
                    data=csv_data,
                    file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with exp_col3:
                json_data = export_clean.to_json(orient="records", force_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Download JSON",
                    data=json_data.encode("utf-8"),
                    file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )

        # ── TAB 2: UTILITY ↔ RENTAL MATCHING ─────────────────────────
        with tab_matched:
            st.markdown(
                "Automatically matches **utility / electricity bills** to their corresponding "
                "**rental invoices** using shared identifiers: Lease ID, Lot No, Vendor Name, TIN No, Account No."
            )

            if not matches:
                st.info(
                    "No utility ↔ rental matches found. This requires utility bills and rental invoices "
                    "with shared Lease ID, Lot No, Vendor Name, or TIN No."
                )
            else:
                st.success(f"🔗 Found **{len(matches)}** utility ↔ rental match(es)")

                for i, m in enumerate(matches, 1):
                    u_row = df.loc[m["utility_idx"]]
                    r_row = df.loc[m["rental_idx"]]
                    match_tags = ", ".join(m["matched_on"])
                    confidence = "🟢 High" if m["score"] >= 6 else ("🟡 Medium" if m["score"] >= 4 else "🟠 Low")

                    with st.container():
                        st.markdown(f"---")
                        st.markdown(f"#### Match {i} — {confidence} confidence")
                        st.caption(f"Matched on: {match_tags}")

                        pair_col1, pair_col2 = st.columns(2)

                        with pair_col1:
                            st.markdown(
                                '<div style="background:#f0fafb; border-left:4px solid #00856e; '
                                'padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.6rem;">'
                                '<strong>⚡ Utility / Electricity Bill</strong></div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Vendor:** {u_row.get('Company Name', '')}")
                            st.markdown(f"**Invoice No:** {u_row.get('Invoice No', '')}")
                            st.markdown(f"**Invoice Date:** {u_row.get('Invoice Date', '')}")
                            st.markdown(f"**Lease ID:** {u_row.get('Lease ID', '')}")
                            st.markdown(f"**Lot No:** {u_row.get('Lot No', '')}")
                            st.markdown(f"**TIN No:** {u_row.get('TIN No', '')}")
                            st.markdown(f"**Amaun Elektrik:** {u_row.get('Amaun Elektrik', '')}")
                            st.markdown(f"**Total (inc. tax):** {u_row.get('Jumlah perlu dibayar' + chr(10) + '(Including tax)', '')}")
                            kwh_b = u_row.get("Kwh Reading Before", "")
                            kwh_a = u_row.get("Kwh Reading After", "")
                            total_u = u_row.get("Current Reading / Total Units", "")
                            if kwh_b or kwh_a:
                                st.markdown(f"**kWh Before → After:** {kwh_b} → {kwh_a}")
                            if total_u:
                                st.markdown(f"**Total Units:** {total_u}")

                        with pair_col2:
                            st.markdown(
                                '<div style="background:#e0f7f0; border-left:4px solid #004d55; '
                                'padding:0.8rem 1rem; border-radius:0 8px 8px 0; margin-bottom:0.6rem;">'
                                '<strong>🏠 Rental / Lease Invoice</strong></div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Vendor:** {r_row.get('Company Name', '')}")
                            st.markdown(f"**Invoice No:** {r_row.get('Invoice No', '')}")
                            st.markdown(f"**Invoice Date:** {r_row.get('Invoice Date', '')}")
                            st.markdown(f"**Lease ID:** {r_row.get('Lease ID', '')}")
                            st.markdown(f"**Lot No:** {r_row.get('Lot No', '')}")
                            st.markdown(f"**TIN No:** {r_row.get('TIN No', '')}")
                            st.markdown(f"**Total (inc. tax):** {r_row.get('Jumlah perlu dibayar' + chr(10) + '(Including tax)', '')}")
                            st.markdown(f"**Description:** {r_row.get('Description', '')}")

                        # Side-by-side comparison table
                        compare_fields = [
                            "Company Name", "TIN No", "Invoice No", "Invoice Date", "Lease ID",
                            "Lot No", "Account No", "Unit No", "Location",
                            "Jumlah perlu dibayar\n(Including tax)",
                        ]
                        compare_rows = []
                        for fld in compare_fields:
                            u_val = str(u_row.get(fld, ""))
                            r_val = str(r_row.get(fld, ""))
                            match_icon = "✅" if (u_val and r_val and u_val.strip() == r_val.strip()) else (
                                "🔶" if (u_val and r_val) else "—"
                            )
                            display_fld = fld.replace("\n", " ")
                            compare_rows.append({
                                "Field": display_fld,
                                "⚡ Utility": u_val,
                                "🏠 Rental": r_val,
                                "Match": match_icon,
                            })
                        with st.expander("📊 Field-by-Field Comparison", expanded=False):
                            st.dataframe(
                                pd.DataFrame(compare_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

            # Show unmatched utility bills
            matched_util_idxs = {m["utility_idx"] for m in matches}
            unmatched_utils = df[
                (df["Types (Inv/CN)"] == "Utility") & (~df.index.isin(matched_util_idxs))
            ]
            if not unmatched_utils.empty:
                st.markdown("---")
                st.markdown("#### ⚠️ Unmatched Utility Bills")
                st.caption("These utility bills could not be matched to any rental invoice.")
                st.dataframe(
                    unmatched_utils[["No", "Company Name", "Invoice No", "Invoice Date",
                                     "Lease ID", "Lot No", "TIN No",
                                     "Jumlah perlu dibayar\n(Including tax)"]].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: BANK MATCHING  –  Enterprise AP Line-Item Reconciliation
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏦 Bank Matching":

    import random
    from datetime import datetime as _dt, timedelta as _td

    # ------------------------------------------------------------------
    # MOCK DATA  (enriched with aging, assignee, investigation status)
    # ------------------------------------------------------------------
    _TODAY = _dt(2026, 2, 24)
    _ASSIGNEES = ["Sarah L.", "Ahmad R.", "Mei Ling C.", "Daniel K."]

    def _mock_date(offset_days):
        d = _TODAY - _td(days=offset_days)
        return d.strftime("%d-%b-%Y")

    def _aging_bucket(days):
        if days <= 30: return "0–30"
        if days <= 60: return "31–60"
        if days <= 90: return "61–90"
        return "90+"

    MOCK_SCENARIOS = {
        "Scenario 1 – Supplier SOA vs Account Ledger (Full Match)": {
            "description": "All line items from the supplier statement match the account ledger perfectly.",
            "supplier": "Acme Corp Sdn Bhd",
            "soa": [
                {"doc_no": "INV-1001", "doc_type": "Invoice",     "date": _mock_date(10), "amount":  10_000},
                {"doc_no": "INV-1002", "doc_type": "Invoice",     "date": _mock_date(8),  "amount":   5_000},
                {"doc_no": "CN-2001",  "doc_type": "Credit Note", "date": _mock_date(5),  "amount":  -1_000},
            ],
            "ledger": [
                {"doc_no": "INV-1001", "doc_type": "Invoice",     "posting_date": _mock_date(9),  "amount":  10_000},
                {"doc_no": "INV-1002", "doc_type": "Invoice",     "posting_date": _mock_date(7),  "amount":   5_000},
                {"doc_no": "CN-2001",  "doc_type": "Credit Note", "posting_date": _mock_date(4),  "amount":  -1_000},
            ],
        },
        "Scenario 2 – Partial Match with Variance": {
            "description": "Some invoices have amount differences or are missing from the ledger.",
            "supplier": "Global Trading Sdn Bhd",
            "soa": [
                {"doc_no": "INV-2001", "doc_type": "Invoice",     "date": _mock_date(45), "amount":  25_000},
                {"doc_no": "INV-2002", "doc_type": "Invoice",     "date": _mock_date(40), "amount":   8_500},
                {"doc_no": "CN-3001",  "doc_type": "Credit Note", "date": _mock_date(35), "amount":  -2_000},
                {"doc_no": "INV-2003", "doc_type": "Invoice",     "date": _mock_date(30), "amount":  15_750},
            ],
            "ledger": [
                {"doc_no": "INV-2001", "doc_type": "Invoice",     "posting_date": _mock_date(44), "amount":  25_000},
                {"doc_no": "INV-2002", "doc_type": "Invoice",     "posting_date": _mock_date(39), "amount":   8_200},
                {"doc_no": "CN-3001",  "doc_type": "Credit Note", "posting_date": _mock_date(34), "amount":  -2_000},
            ],
        },
        "Scenario 3 – Multi-Supplier Reconciliation": {
            "description": "Complex reconciliation across multiple document types with partial matches.",
            "supplier": "Premier Supplies Sdn Bhd",
            "soa": [
                {"doc_no": "INV-5001", "doc_type": "Invoice",     "date": _mock_date(95), "amount":  42_800},
                {"doc_no": "INV-5002", "doc_type": "Invoice",     "date": _mock_date(75), "amount":  18_350},
                {"doc_no": "INV-5003", "doc_type": "Invoice",     "date": _mock_date(55), "amount":   6_900},
                {"doc_no": "CN-5001",  "doc_type": "Credit Note", "date": _mock_date(40), "amount":  -3_200},
                {"doc_no": "INV-5004", "doc_type": "Invoice",     "date": _mock_date(20), "amount":  31_500},
                {"doc_no": "DN-5001",  "doc_type": "Debit Note",  "date": _mock_date(10), "amount":   1_200},
            ],
            "ledger": [
                {"doc_no": "INV-5001", "doc_type": "Invoice",     "posting_date": _mock_date(93), "amount":  42_800},
                {"doc_no": "INV-5002", "doc_type": "Invoice",     "posting_date": _mock_date(73), "amount":  18_350},
                {"doc_no": "INV-5003", "doc_type": "Invoice",     "posting_date": _mock_date(53), "amount":   6_500},
                {"doc_no": "CN-5001",  "doc_type": "Credit Note", "posting_date": _mock_date(38), "amount":  -3_200},
                {"doc_no": "INV-5004", "doc_type": "Invoice",     "posting_date": _mock_date(18), "amount":  31_500},
                {"doc_no": "DN-5001",  "doc_type": "Debit Note",  "posting_date": _mock_date(8),  "amount":   1_200},
            ],
        },
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fmt_amt(v):
        if v is None:
            return '<span style="color:#a0aec0;">—</span>'
        if v < 0:
            return f'<span class="neg-amount">({abs(v):,.0f})</span>'
        return f"{v:,.0f}"

    def _parse_mock_date(d_str):
        try:
            return _dt.strptime(d_str, "%d-%b-%Y")
        except Exception:
            return _TODAY

    def _build_matching_results(soa_items, ledger_items):
        """Build enriched matching results with aging & assignment."""
        ledger_map = {r["doc_no"]: r for r in ledger_items}
        soa_map = {r["doc_no"]: r for r in soa_items}
        results = []
        random.seed(42)  # deterministic for POC

        # Items from SOA
        for s in soa_items:
            doc = s["doc_no"]
            aging_days = (_TODAY - _parse_mock_date(s["date"])).days
            base = {
                "doc_no": doc,
                "doc_type": s["doc_type"],
                "soa_amount": s["amount"],
                "soa_date": s["date"],
                "aging_days": aging_days,
                "aging_bucket": _aging_bucket(aging_days),
                "assigned_to": random.choice(_ASSIGNEES),
                "last_updated": _mock_date(random.randint(0, 5)),
            }
            if doc in ledger_map:
                l = ledger_map[doc]
                base["ledger_amount"] = l["amount"]
                base["ledger_date"] = l["posting_date"]
                base["date_diff"] = abs((_parse_mock_date(s["date"]) - _parse_mock_date(l["posting_date"])).days)
                if s["amount"] == l["amount"]:
                    base["variance"] = 0
                    base["status"] = "Match"
                    base["investigation"] = "Approved"
                else:
                    base["variance"] = s["amount"] - l["amount"]
                    base["status"] = "Amount Mismatch"
                    base["investigation"] = random.choice(["Under Investigation", "Resolved"])
            else:
                base["ledger_amount"] = None
                base["ledger_date"] = None
                base["date_diff"] = None
                base["variance"] = None
                base["status"] = "Missing in Ledger"
                base["investigation"] = "Under Investigation"
            results.append(base)

        # Items in Ledger but not in SOA
        for l in ledger_items:
            if l["doc_no"] not in soa_map:
                aging_days = (_TODAY - _parse_mock_date(l["posting_date"])).days
                results.append({
                    "doc_no": l["doc_no"],
                    "doc_type": l["doc_type"],
                    "soa_amount": None,
                    "soa_date": None,
                    "ledger_amount": l["amount"],
                    "ledger_date": l["posting_date"],
                    "date_diff": None,
                    "aging_days": aging_days,
                    "aging_bucket": _aging_bucket(aging_days),
                    "variance": None,
                    "status": "Missing in SOA",
                    "investigation": "Under Investigation",
                    "assigned_to": random.choice(_ASSIGNEES),
                    "last_updated": _mock_date(random.randint(0, 5)),
                })
        return results

    _STATUS_BADGE = {
        "Match":              '<span class="lim-badge lim-badge-match">✔ Match</span>',
        "Amount Mismatch":    '<span class="lim-badge lim-badge-mismatch">⚠ Amount Mismatch</span>',
        "Missing in Ledger":  '<span class="lim-badge lim-badge-missing-ledger">✖ Missing in Ledger</span>',
        "Missing in SOA":     '<span class="lim-badge lim-badge-missing-soa">✖ Missing in SOA</span>',
    }
    _INV_BADGE = {
        "Under Investigation": '<span class="lim-badge lim-badge-investigating">🔍 Investigating</span>',
        "Resolved":            '<span class="lim-badge lim-badge-resolved">✔ Resolved</span>',
        "Approved":            '<span class="lim-badge lim-badge-approved">✔ Approved</span>',
    }

    # ══════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════
    st.markdown("""
    <div class="lim-header">
        <div class="lim-header-icon">📊</div>
        <div>
            <h2>AP Line-Item Reconciliation</h2>
            <p>Supplier statement vs Account ledger &mdash; automated matching, variance analysis &amp; exception management.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scenario selector ────────────────────────────────────
    sel_c, info_c = st.columns([3, 5])
    with sel_c:
        scenario_name = st.selectbox("Scenario", list(MOCK_SCENARIOS.keys()), label_visibility="collapsed")
    scenario = MOCK_SCENARIOS[scenario_name]
    with info_c:
        st.caption(f"**{scenario['supplier']}** — {scenario['description']}")

    soa_items = scenario["soa"]
    ledger_items = scenario["ledger"]
    match_results = _build_matching_results(soa_items, ledger_items)

    # Aggregates
    soa_total = sum(r["amount"] for r in soa_items)
    led_total = sum(r["amount"] for r in ledger_items)
    net_variance = soa_total - led_total
    n_total = len(match_results)
    n_match = sum(1 for r in match_results if r["status"] == "Match")
    n_mismatch = sum(1 for r in match_results if r["status"] == "Amount Mismatch")
    n_missing = sum(1 for r in match_results if r["status"].startswith("Missing"))
    match_rate = (n_match / n_total * 100) if n_total else 0
    total_mismatch_amt = sum(abs(r["variance"]) for r in match_results if r["status"] == "Amount Mismatch" and r["variance"])
    overdue_variance = sum(abs(r["variance"] or 0) for r in match_results if r["status"] != "Match" and r["aging_days"] > 30)

    # KPI risk colour
    def _risk_color(val, baseline=0, minor_thresh=500):
        if val == baseline:
            return "green"
        if abs(val - baseline) <= minor_thresh:
            return "amber"
        return "red"

    # ══════════════════════════════════════════════════════════
    # KPI CARDS  (6 metrics)
    # ══════════════════════════════════════════════════════════
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    _kpi_data = [
        (k1, f"{soa_total:,.0f}", "Total Exposure (SOA)", "Total amount claimed by supplier", "navy"),
        (k2, f"{led_total:,.0f}", "Ledger Total", "Sum posted in your books", "blue"),
        (k3, f"{_fmt_amt(net_variance)}", "Net Variance", "SOA − Ledger difference", _risk_color(net_variance)),
        (k4, f"{match_rate:.0f}%", "Match Rate", "Percentage of fully matched items", "green" if match_rate == 100 else ("amber" if match_rate >= 80 else "red")),
        (k5, f"{total_mismatch_amt:,.0f}", "Mismatch Amount", "Sum of absolute variances", "red" if total_mismatch_amt else "green"),
        (k6, f"{overdue_variance:,.0f}", "Overdue Variance", "Unresolved variance > 30 days", "red" if overdue_variance else "green"),
    ]
    for col, val, label, tooltip, color in _kpi_data:
        with col:
            st.markdown(f"""
            <div class="lim-kpi-card lim-kpi-{color}">
                <div class="kpi-border-top"></div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-tooltip">{tooltip}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════════════════════
    # SIDE-BY-SIDE SOURCE TABLES
    # ══════════════════════════════════════════════════════════
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.markdown('<div class="lim-section-title">📄 Supplier SOA</div>', unsafe_allow_html=True)
        soa_hdr = "".join(f"<th>{h}</th>" for h in ["Doc No", "Doc Type", "Date", "Amount"])
        soa_body = ""
        for r in soa_items:
            soa_body += f'<tr><td>{r["doc_no"]}</td><td>{r["doc_type"]}</td><td>{r["date"]}</td><td class="amount-col">{_fmt_amt(r["amount"])}</td></tr>'
        soa_body += f'<tr style="background:#e0f7f9;font-weight:700;"><td colspan="3">Total</td><td class="amount-col">{_fmt_amt(soa_total)}</td></tr>'
        st.markdown(f'<table class="lim-table"><thead><tr>{soa_hdr}</tr></thead><tbody>{soa_body}</tbody></table>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="lim-section-title">📒 Account Ledger</div>', unsafe_allow_html=True)
        led_hdr = "".join(f"<th>{h}</th>" for h in ["Doc No", "Doc Type", "Posting Date", "Amount"])
        led_body = ""
        for r in ledger_items:
            led_body += f'<tr><td>{r["doc_no"]}</td><td>{r["doc_type"]}</td><td>{r["posting_date"]}</td><td class="amount-col">{_fmt_amt(r["amount"])}</td></tr>'
        led_body += f'<tr style="background:#e0f7f0;font-weight:700;"><td colspan="3">Total</td><td class="amount-col">{_fmt_amt(led_total)}</td></tr>'
        st.markdown(f'<table class="lim-table lim-table-ledger"><thead><tr>{led_hdr}</tr></thead><tbody>{led_body}</tbody></table>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # FILTER BAR + EXCEPTION TOGGLE
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="lim-section-title">🔎 Filters</div>', unsafe_allow_html=True)
    with st.container():
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 1])
        with fc1:
            flt_status = st.multiselect(
                "Match Status",
                ["Match", "Amount Mismatch", "Missing in Ledger", "Missing in SOA"],
                default=[],
                placeholder="All statuses",
            )
        with fc2:
            flt_aging = st.multiselect(
                "Aging Bucket",
                ["0–30", "31–60", "61–90", "90+"],
                default=[],
                placeholder="All buckets",
            )
        with fc3:
            flt_investigation = st.multiselect(
                "Investigation Status",
                ["Under Investigation", "Resolved", "Approved"],
                default=[],
                placeholder="All",
            )
        with fc4:
            flt_show = st.toggle("Show Exceptions Only", value=True, help="When ON, only mismatched / missing items are shown.")
        with fc5:
            if st.button("↻ Reset", use_container_width=True):
                st.rerun()

    # Apply filters
    filtered = match_results
    if flt_show:
        filtered = [r for r in filtered if r["status"] != "Match"]
    if flt_status:
        filtered = [r for r in filtered if r["status"] in flt_status]
    if flt_aging:
        filtered = [r for r in filtered if r["aging_bucket"] in flt_aging]
    if flt_investigation:
        filtered = [r for r in filtered if r["investigation"] in flt_investigation]

    # Sort: exceptions first
    _status_order = {"Amount Mismatch": 0, "Missing in Ledger": 1, "Missing in SOA": 2, "Match": 3}
    filtered.sort(key=lambda r: (_status_order.get(r["status"], 9), -abs(r["variance"] or 0)))

    # ══════════════════════════════════════════════════════════
    # LINE-ITEM MATCHING RESULTS TABLE
    # ══════════════════════════════════════════════════════════
    st.markdown(f'<div class="lim-section-title">🎯 Line-Item Matching Results &nbsp;<span style="font-weight:400;color:#5a8a8f;font-size:0.82rem;">({len(filtered)} of {n_total} items)</span></div>', unsafe_allow_html=True)

    _cols = ["Doc No", "Type", "SOA Amt", "Ledger Amt", "Variance", "Aging", "Bucket", "Assigned To", "Updated", "Status", "Investigation"]
    res_hdr = "".join(f"<th>{c}</th>" for c in _cols)
    res_body = ""
    for r in filtered:
        var_td = ""
        if r["variance"] is not None and r["variance"] != 0:
            var_td = f'<td class="amount-col" style="color:#EE2D25;font-weight:800;" title="SOA {r["soa_amount"]:,.0f} − Ledger {r["ledger_amount"]:,.0f}">{_fmt_amt(r["variance"])}</td>'
        elif r["variance"] == 0:
            var_td = '<td class="amount-col" style="color:#00856e;">0</td>'
        else:
            var_td = '<td class="amount-col">' + _fmt_amt(None) + '</td>'

        row_bg = ""
        if r["status"] == "Amount Mismatch":
            row_bg = ' style="background:#fff8e6;"'
        elif r["status"].startswith("Missing"):
            row_bg = ' style="background:#fde8e7;"'

        res_body += (
            f'<tr{row_bg}>'
            f'<td><strong>{r["doc_no"]}</strong></td>'
            f'<td>{r["doc_type"]}</td>'
            f'<td class="amount-col">{_fmt_amt(r["soa_amount"])}</td>'
            f'<td class="amount-col">{_fmt_amt(r["ledger_amount"])}</td>'
            f'{var_td}'
            f'<td style="text-align:center;">{r["aging_days"]}d</td>'
            f'<td>{r["aging_bucket"]}</td>'
            f'<td>{r["assigned_to"]}</td>'
            f'<td>{r["last_updated"]}</td>'
            f'<td>{_STATUS_BADGE.get(r["status"], r["status"])}</td>'
            f'<td>{_INV_BADGE.get(r["investigation"], r["investigation"])}</td>'
            f'</tr>'
        )
    st.markdown(f'<table class="lim-table lim-table-result"><thead><tr>{res_hdr}</tr></thead><tbody>{res_body}</tbody></table>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # DRILL-DOWN DETAIL PANEL
    # ══════════════════════════════════════════════════════════
    st.markdown("")
    st.markdown('<div class="lim-section-title">🔍 Document Drill-Down</div>', unsafe_allow_html=True)
    doc_options = [r["doc_no"] for r in match_results]
    # Default to first exception if any
    exception_docs = [r["doc_no"] for r in match_results if r["status"] != "Match"]
    default_idx = doc_options.index(exception_docs[0]) if exception_docs else 0

    dd_doc = st.selectbox("Select document to inspect", doc_options, index=default_idx, label_visibility="collapsed")
    dd_item = next((r for r in match_results if r["doc_no"] == dd_doc), None)

    if dd_item:
        st.markdown('<div class="lim-drill-panel">', unsafe_allow_html=True)

        dl_left, dl_mid, dl_right = st.columns(3)
        with dl_left:
            st.markdown("**📄 SOA Details**")
            if dd_item["soa_amount"] is not None:
                st.markdown(f"""
                <div class="lim-drill-field"><div class="dl-label">Doc No</div><div class="dl-value">{dd_item['doc_no']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">Type</div><div class="dl-value">{dd_item['doc_type']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">SOA Date</div><div class="dl-value">{dd_item['soa_date']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">Amount</div><div class="dl-value">{_fmt_amt(dd_item['soa_amount'])}</div></div>
                """, unsafe_allow_html=True)
            else:
                st.caption("_Not present in supplier SOA._")

        with dl_mid:
            st.markdown("**📒 Ledger Details**")
            if dd_item["ledger_amount"] is not None:
                amt_class = "dl-highlight" if dd_item["status"] == "Amount Mismatch" else "dl-ok"
                st.markdown(f"""
                <div class="lim-drill-field"><div class="dl-label">Doc No</div><div class="dl-value">{dd_item['doc_no']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">Type</div><div class="dl-value">{dd_item['doc_type']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">Posting Date</div><div class="dl-value">{dd_item['ledger_date']}</div></div>
                <div class="lim-drill-field"><div class="dl-label">Amount</div><div class="dl-value {amt_class}">{_fmt_amt(dd_item['ledger_amount'])}</div></div>
                """, unsafe_allow_html=True)
            else:
                st.caption("_Not found in account ledger._")

        with dl_right:
            st.markdown("**📐 Comparison**")
            var_display = _fmt_amt(dd_item["variance"]) if dd_item["variance"] is not None else "N/A"
            var_class = "dl-highlight" if (dd_item["variance"] and dd_item["variance"] != 0) else "dl-ok"
            date_diff_display = f'{dd_item["date_diff"]} days' if dd_item["date_diff"] is not None else "N/A"
            st.markdown(f"""
            <div class="lim-drill-field"><div class="dl-label">Variance</div><div class="dl-value {var_class}">{var_display}</div></div>
            <div class="lim-drill-field"><div class="dl-label">Date Difference</div><div class="dl-value">{date_diff_display}</div></div>
            <div class="lim-drill-field"><div class="dl-label">Aging</div><div class="dl-value">{dd_item['aging_days']} days ({dd_item['aging_bucket']})</div></div>
            <div class="lim-drill-field"><div class="dl-label">Status</div><div class="dl-value">{_STATUS_BADGE.get(dd_item['status'], dd_item['status'])}</div></div>
            <div class="lim-drill-field"><div class="dl-label">Investigation</div><div class="dl-value">{_INV_BADGE.get(dd_item['investigation'], dd_item['investigation'])}</div></div>
            """, unsafe_allow_html=True)

        # Action row
        st.markdown("</div>", unsafe_allow_html=True)
        act1, act2, act3, act4 = st.columns([3, 2, 2, 5])
        with act1:
            st.text_area("Add Comment", placeholder="Enter investigation notes…", height=68, key="drill_comment", label_visibility="collapsed")
        with act2:
            st.button("📎 Attach Document", use_container_width=True, key="drill_attach")
        with act3:
            st.button("✅ Mark Resolved", use_container_width=True, type="primary", key="drill_resolve")

    st.markdown("")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # RECONCILIATION SUMMARY BAR + PROGRESS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="lim-section-title">📋 Reconciliation Summary</div>', unsafe_allow_html=True)

    pbar_color = "#00856e" if match_rate == 100 else ("#d68000" if match_rate >= 70 else "#EE2D25")
    st.markdown(f"""
    <div class="lim-recon-bar">
        <div class="lim-recon-row"><span class="recon-label">SOA Total (Supplier Claimed)</span><span class="recon-value">{_fmt_amt(soa_total)}</span></div>
        <div class="lim-recon-divider"></div>
        <div class="lim-recon-row"><span class="recon-label">Ledger Total (Your Books)</span><span class="recon-value">{_fmt_amt(led_total)}</span></div>
        <div class="lim-recon-divider"></div>
        <div class="lim-recon-row"><span class="recon-label">Net Variance</span><span class="recon-value" style="color:{'#00856e' if net_variance == 0 else '#EE2D25'};">{_fmt_amt(net_variance)}</span></div>
        <div class="lim-recon-divider"></div>
        <div class="lim-recon-row">
            <span class="recon-label">Match Progress</span>
            <span class="recon-value" style="color:{pbar_color};">{n_match}/{n_total} items ({match_rate:.0f}%)</span>
        </div>
        <div class="lim-progress-track">
            <div class="lim-progress-fill" style="width:{match_rate:.0f}%;background:{pbar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ══════════════════════════════════════════════════════════
    # AGING VISUALIZATION
    # ══════════════════════════════════════════════════════════
    exceptions = [r for r in match_results if r["status"] != "Match"]
    if exceptions:
        st.markdown('<div class="lim-section-title">📅 Exception Aging Distribution</div>', unsafe_allow_html=True)

        bucket_counts = {"0–30": 0, "31–60": 0, "61–90": 0, "90+": 0}
        for r in exceptions:
            bucket_counts[r["aging_bucket"]] = bucket_counts.get(r["aging_bucket"], 0) + 1

        df_aging = pd.DataFrame({
            "Aging Bucket": list(bucket_counts.keys()),
            "Exceptions": list(bucket_counts.values()),
        })
        ag_left, ag_right = st.columns([3, 5])
        with ag_left:
            st.dataframe(df_aging, use_container_width=True, hide_index=True)
        with ag_right:
            st.bar_chart(df_aging.set_index("Aging Bucket"), color="#00A0AF", height=220)

    st.markdown("")

    # ══════════════════════════════════════════════════════════
    # VARIANCE ANALYSIS (detailed)
    # ══════════════════════════════════════════════════════════
    variances = [r for r in match_results if r["status"] in ("Amount Mismatch", "Missing in Ledger", "Missing in SOA")]
    if variances:
        st.markdown('<div class="lim-section-title">⚠️ Variance Analysis</div>', unsafe_allow_html=True)
        for v in variances:
            if v["status"] == "Amount Mismatch":
                st.markdown(f"""
                <div class="lim-variance-row">
                    <strong>{v['doc_no']}</strong> &mdash;
                    SOA: <strong>{_fmt_amt(v['soa_amount'])}</strong> &nbsp;|&nbsp;
                    Ledger: <strong>{_fmt_amt(v['ledger_amount'])}</strong> &nbsp;|&nbsp;
                    Variance: <strong style="color:#EE2D25;">{_fmt_amt(v['variance'])}</strong>
                    &nbsp;&nbsp;|&nbsp;&nbsp;Aging: <strong>{v['aging_days']}d</strong> ({v['aging_bucket']})
                    &nbsp;&nbsp;<em style="color:#5a8a8f;">→ Assigned to {v['assigned_to']}</em>
                </div>""", unsafe_allow_html=True)
            elif v["status"] == "Missing in Ledger":
                st.markdown(f"""
                <div class="lim-variance-row" style="border-left-color:#EE2D25;background:#fde8e7;">
                    <strong>{v['doc_no']}</strong> &mdash;
                    SOA: <strong>{_fmt_amt(v['soa_amount'])}</strong> &nbsp;|&nbsp;
                    Ledger: <strong style="color:#8ab0b5;">Not found</strong>
                    &nbsp;&nbsp;|&nbsp;&nbsp;Aging: <strong>{v['aging_days']}d</strong>
                    &nbsp;&nbsp;<em style="color:#5a8a8f;">→ Document in SOA but missing from ledger. Assigned to {v['assigned_to']}</em>
                </div>""", unsafe_allow_html=True)
            else:  # Missing in SOA
                st.markdown(f"""
                <div class="lim-variance-row" style="border-left-color:#00A0AF;background:#e0f7f9;">
                    <strong>{v['doc_no']}</strong> &mdash;
                    Ledger: <strong>{_fmt_amt(v['ledger_amount'])}</strong> &nbsp;|&nbsp;
                    SOA: <strong style="color:#8ab0b5;">Not found</strong>
                    &nbsp;&nbsp;|&nbsp;&nbsp;Aging: <strong>{v['aging_days']}d</strong>
                    &nbsp;&nbsp;<em style="color:#5a8a8f;">→ Document posted but not in supplier statement. Assigned to {v['assigned_to']}</em>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════
    st.markdown("")
    exp1, exp2, _ = st.columns([2, 2, 8])
    with exp1:
        df_exp = pd.DataFrame([{
            "Doc No": r["doc_no"], "Type": r["doc_type"],
            "SOA Amount": r["soa_amount"], "Ledger Amount": r["ledger_amount"],
            "Variance": r["variance"], "Status": r["status"],
            "Aging Days": r["aging_days"], "Aging Bucket": r["aging_bucket"],
            "Assigned To": r["assigned_to"], "Investigation": r["investigation"],
            "Last Updated": r["last_updated"],
        } for r in match_results])
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Export Full Results (CSV)", data=csv, file_name="ap_reconciliation.csv", mime="text/csv")
    with exp2:
        lines = [
            f"AP Line-Item Reconciliation  –  {scenario['supplier']}",
            f"Generated: {_TODAY.strftime('%d %b %Y')}",
            "=" * 60, "",
            f"{'Doc No':14s}  {'SOA':>12s}  {'Ledger':>12s}  {'Variance':>10s}  {'Status':20s}  {'Aging':>5s}  {'Investigation'}",
            "-" * 100,
        ]
        for r in match_results:
            sv = f"{r['soa_amount']:,.0f}" if r['soa_amount'] is not None else "-"
            lv = f"{r['ledger_amount']:,.0f}" if r['ledger_amount'] is not None else "-"
            vv = f"{r['variance']:,.0f}" if r['variance'] is not None else "-"
            lines.append(f"{r['doc_no']:14s}  {sv:>12s}  {lv:>12s}  {vv:>10s}  {r['status']:20s}  {r['aging_days']:>3d}d  {r['investigation']}")
        lines += ["", "-" * 100, f"SOA Total:      {soa_total:>12,.0f}", f"Ledger Total:   {led_total:>12,.0f}",
                   f"Net Variance:   {net_variance:>12,.0f}", f"Match Rate:     {match_rate:>11.0f}%"]
        st.download_button("📄  Export Report (TXT)", data="\n".join(lines), file_name="ap_reconciliation_report.txt", mime="text/plain")
