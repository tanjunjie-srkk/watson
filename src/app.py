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

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d5a87 50%, #1e6f5c 100%);
        padding: 1.2rem 2rem; border-radius: 10px; margin-bottom: 1.5rem; color: white;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0; font-weight: 700; }
    .main-header p { color: #c8dae8; font-size: 0.95rem; margin: 0.3rem 0 0 0; }
    .pipeline-step {
        background: #f7fafc; border-left: 4px solid #2d5a87;
        padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 0.8rem;
    }
    .confidence-high { color: #38a169; font-weight: 600; }
    .confidence-medium { color: #d69e2e; font-weight: 600; }
    .confidence-low { color: #e53e3e; font-weight: 600; }
    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stSidebar"] .stCaption p { color: #94a3b8 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
    .sidebar-logo {
        text-align: center; padding: 1.2rem 0 0.6rem 0;
    }
    .sidebar-logo-icon {
        font-size: 2.2rem; display: block; margin-bottom: 0.3rem;
    }
    .sidebar-logo-text {
        font-size: 1.15rem; font-weight: 700; color: #f1f5f9;
        letter-spacing: 0.02em;
    }
    .sidebar-logo-sub {
        font-size: 0.72rem; color: #64748b; letter-spacing: 0.06em;
        text-transform: uppercase; margin-top: 2px;
    }
    .sidebar-section {
        font-size: 0.68rem; font-weight: 600; color: #475569;
        text-transform: uppercase; letter-spacing: 0.1em;
        padding: 0.8rem 0 0.4rem 0.5rem;
    }
    .sidebar-about-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.3rem;
    }
    .sidebar-about-title {
        font-size: 0.7rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;
    }
    .sidebar-about-item {
        display: flex; align-items: flex-start; gap: 0.5rem;
        padding: 0.35rem 0; font-size: 0.8rem; color: #94a3b8;
    }
    .sidebar-about-item strong { color: #cbd5e1; }
    .sidebar-badge {
        display: inline-block; font-size: 0.62rem; font-weight: 600;
        padding: 2px 8px; border-radius: 20px;
        background: rgba(56,161,105,0.15); color: #4ade80;
        letter-spacing: 0.03em;
    }
    .sidebar-footer {
        text-align: center; padding: 0.5rem 0; font-size: 0.72rem; color: #475569;
    }
    .info-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem; margin-bottom: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .info-card .card-title { font-weight: 700; color: #1a365d; margin-bottom: 0.3rem; }
    .info-card .card-subtitle { font-size: 0.8rem; color: #718096; }

    /* Report card styles */
    .doc-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 0; margin-bottom: 0.75rem; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); transition: box-shadow 0.2s;
    }
    .doc-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .doc-card-header {
        display: flex; align-items: center; padding: 0.8rem 1.2rem;
        gap: 1rem; cursor: pointer;
    }
    .doc-card-num {
        font-weight: 700; font-size: 0.85rem; color: #718096;
        min-width: 32px; text-align: center;
    }
    .doc-card-type {
        font-size: 0.7rem; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; text-transform: uppercase; letter-spacing: 0.03em;
    }
    .doc-card-company { font-weight: 600; color: #1a365d; font-size: 0.95rem; flex: 1; }
    .doc-card-detail { color: #4a5568; font-size: 0.85rem; }
    .doc-card-amount { font-weight: 700; color: #1a365d; font-size: 1rem; white-space: nowrap; }
    .doc-card-match {
        font-size: 0.72rem; padding: 2px 8px; border-radius: 10px;
        font-weight: 600; white-space: nowrap;
    }
    .doc-field-grid {
        display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 0.6rem; padding: 0.8rem 1.2rem;
    }
    .doc-field { padding: 0.4rem 0; }
    .doc-field-label { font-size: 0.7rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
    .doc-field-value { font-size: 0.88rem; color: #2d3748; font-weight: 500; word-break: break-word; }
    .doc-status-verified { border-left: 4px solid #38a169; }
    .doc-status-rejected { border-left: 4px solid #e53e3e; opacity: 0.6; }
    .doc-status-pending { border-left: 4px solid #e2e8f0; }
    .doc-card-status { font-size: 0.8rem; white-space: nowrap; color: #718096; }
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
    displayed = [(label, str(data[key])) for key, label in core_fields if data.get(key) and str(data.get(key)) != "null"]

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
    totals = [(lbl, data[k]) for k, lbl in total_fields if data.get(k)]
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
        "Jumlah perlu dibayar\n(Including tax)": _safe(data, "grand_total") or _safe(data, "total_amount"),
        "Amaun Elektrik": _electricity_amount(data),
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
        files = sorted(extraction_dir.glob("*_extracted*.json"))
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
    <h1>📄 Watson Document Intelligence Platform</h1>
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
        '<div class="sidebar-logo-text">Watson Intelligence</div>'
        '<div class="sidebar-logo-sub">Document AI Platform</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Navigation ────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        [
            "🏠 Dashboard",
            "📤 Document Processing",
            "🔍 OCR Viewer",
            "📊 Extraction Viewer",
            "📋 Report Format",
            "🏦 Bank Matching",
        ],
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
        f'<span style="color:#475569;font-size:0.7rem;">Powered by Azure OpenAI &bull; {datetime.now().strftime("%d %b %Y")}</span>'
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


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    extraction_dir = Path(__file__).resolve().parent / "extraction_output"
    ocr_dir = Path(__file__).resolve().parent / "ocr_output"
    docs_dir = Path(__file__).resolve().parent / "docs"

    num_extracted = len(list(extraction_dir.glob("*_extracted.json"))) if extraction_dir.exists() else 0
    num_ocr = len(list(ocr_dir.glob("*.json"))) if ocr_dir.exists() else 0
    num_pdfs = len(list(docs_dir.glob("*.pdf"))) if docs_dir.exists() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Source PDFs", num_pdfs)
    c2.metric("🔍 OCR Processed", num_ocr)
    c3.metric("📊 Extractions", num_extracted)
    c4.metric("🏷️ Doc Types", len(AGENT_REGISTRY))

    st.divider()
    st.markdown("### 🔄 Processing Pipeline")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        <div class="pipeline-step"><strong>Step 1 — PDF to Images</strong><br/><span style="color:#718096;">Upload PDF → Convert each page to high-resolution PNG (300 DPI)</span></div>
        <div class="pipeline-step"><strong>Step 2 — AI-Powered OCR</strong><br/><span style="color:#718096;">GPT vision model reads every character with confidence scoring</span></div>
        <div class="pipeline-step"><strong>Step 3 — Document Classification</strong><br/><span style="color:#718096;">AI classifies: Invoice, Utility Bill, Bank Statement, Travel, etc.</span></div>
        <div class="pipeline-step"><strong>Step 4 — Structured Extraction</strong><br/><span style="color:#718096;">Type-specific agent extracts vendor, amounts, line items, dates</span></div>
        <div class="pipeline-step"><strong>Step 5 — Bank Matching</strong><br/><span style="color:#718096;">Reconcile extracted invoices vs bank entries (exact + near matches)</span></div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("#### 📋 Supported Document Types")
        for dtype, (icon, desc) in {
            "commercial_invoice": ("🧾", "Product invoices, PO numbers, shipping"),
            "credit_note": ("💳", "Credit notes, refund documents"),
            "travel": ("✈️", "Flight tickets, travel invoices"),
            "rental": ("🏢", "Mall rent, lease, service charges"),
            "hotel": ("🏨", "Hotel folios, room charges"),
            "utility": ("⚡", "Electricity, water, gas, telecom"),
            "soa": ("📑", "Statements of account, aging reports"),
            "bank_statement": ("🏦", "Bank transaction listings"),
        }.items():
            st.markdown(f"**{icon} {dtype.replace('_',' ').title()}** — {desc}")

    st.divider()
    st.markdown("### 📂 Recently Processed Documents")

    if extraction_dir.exists():
        files = sorted(extraction_dir.glob("*_extracted.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
        if files:
            cols = st.columns(4)
            for i, f in enumerate(files):
                with cols[i % 4]:
                    d = load_json_file(f)
                    dt = d.get("document_type", "Unknown") if isinstance(d, dict) else "Unknown"
                    vn = d.get("vendor_name") or "N/A" if isinstance(d, dict) else "N/A"
                    gt = (d.get("grand_total") or d.get("total_amount") or "N/A") if isinstance(d, dict) else "N/A"
                    st.markdown(f"""<div class="info-card">
                        <div class="card-title">{f.stem}</div>
                        <div class="card-subtitle">{dt}</div>
                        <div style="font-size:0.85rem;margin-top:0.4rem;"><strong>Vendor:</strong> {str(vn)[:35]}<br/><strong>Total:</strong> {gt}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("No extracted documents yet. Go to **Document Processing** to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DOCUMENT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📤 Document Processing":

    st.markdown("### 📤 Document Processing Pipeline")
    st.markdown("Upload a PDF to run the full AI pipeline: **PDF → Images → OCR → Classification → Extraction**")

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"],
            help="Invoices, Utility Bills, Bank Statements, Travel, Rental, SOA, etc.")
    with col_opt:
        force_type = st.selectbox("Force document type (optional)", ["Auto-detect"] + list(AGENT_REGISTRY.keys()))
        ocr_mode = st.radio("OCR Mode", ["Batch (all pages)", "Per-page"], index=0)

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.getvalue())
            st.markdown(f"**📄 Uploaded:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

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

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button("⬇️ Download OCR JSON",
                        data=json.dumps(ocr_parsed, ensure_ascii=False, indent=2),
                        file_name=f"{pdf_path.stem}_ocr.json", mime="application/json")
                with dc2:
                    st.download_button("⬇️ Download Extraction JSON",
                        data=json.dumps(extracted, ensure_ascii=False, indent=2),
                        file_name=f"{pdf_path.stem}_extracted.json", mime="application/json")


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
    st.markdown("View structured financial data extracted from documents.")

    extraction_dir = Path(__file__).resolve().parent / "extraction_output"
    if extraction_dir.exists():
        ext_files = sorted(extraction_dir.glob("*_extracted.json"))
        if ext_files:
            sel = st.selectbox("Select extraction file", ext_files, format_func=lambda p: p.name)
            if sel:
                data = load_json_file(sel)
                if isinstance(data, dict):
                    tab_d, tab_j = st.tabs(["📊 Structured View", "📝 Raw JSON"])
                    with tab_d:
                        display_extraction_result(data, data.get("document_type", "Unknown"))
                    with tab_j:
                        st.json(data)
                else:
                    st.json(data)
        else:
            st.info("No extraction files found.")
    else:
        st.info("Extraction output directory not found.")


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

    if df.empty:
        st.info("No extraction files found. Process some documents first.")
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
                "Inv": ("🧾", "#e8f0fe", "#1a56db"),
                "Utility": ("⚡", "#fef9e7", "#b7791f"),
                "Rental": ("🏠", "#e6ffed", "#276749"),
                "Hotel": ("🏨", "#fce7f3", "#9d174d"),
                "Travel": ("✈️", "#eff6ff", "#1e40af"),
                "SOA": ("📑", "#f3e8ff", "#6b21a8"),
                "CN": ("📌", "#fef2f2", "#b91c1c"),
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
                    conf_colors = {"High": "#38a169", "Medium": "#d69e2e", "Low": "#e53e3e"}
                    mc = conf_colors.get(conf, "#718096")
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
                st.markdown(card_html, unsafe_allow_html=True)

                # ── Expandable detail section ─────────────────────────
                with st.expander(f"View details — #{row_no}", expanded=False):
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

                    # ── Key fields grid ────────────────────────────────
                    key_fields = [
                        ("Invoice No", inv_no),
                        ("Invoice Date", inv_date),
                        ("TIN No", row.get("TIN No", "")),
                        ("Total (inc. tax)", total),
                    ]
                    grid_html = '<div class="doc-field-grid">'
                    for lbl, val in key_fields:
                        v = val if val else "—"
                        grid_html += f'<div class="doc-field"><div class="doc-field-label">{lbl}</div><div class="doc-field-value">{v}</div></div>'
                    grid_html += '</div>'
                    st.markdown(grid_html, unsafe_allow_html=True)

                    # ── Location & IDs ─────────────────────────────────
                    loc_pairs = [
                        ("Lot No", row.get("Lot No", "")),
                        ("Location", row.get("Location", "")),
                        ("Account No", row.get("Account No", "")),
                        ("Lease ID", row.get("Lease ID", "")),
                        ("Unit No", row.get("Unit No", "")),
                        ("No. Akaun", row.get("No. Akaun", "")),
                    ]
                    filled_loc = [(l, v) for l, v in loc_pairs if v]
                    if filled_loc:
                        loc_html = '<div class="doc-field-grid">'
                        for lbl, val in filled_loc:
                            loc_html += f'<div class="doc-field"><div class="doc-field-label">{lbl}</div><div class="doc-field-value">{val}</div></div>'
                        loc_html += '</div>'
                        st.markdown(loc_html, unsafe_allow_html=True)

                    # ── Electricity details ────────────────────────────
                    elec_amt = row.get("Amaun Elektrik", "")
                    kwh_b = row.get("Kwh Reading Before", "")
                    kwh_a = row.get("Kwh Reading After", "")
                    total_u = row.get("Current Reading / Total Units", "")
                    if elec_amt or kwh_b or kwh_a:
                        st.markdown("**⚡ Electricity Details**")
                        elec_pairs = [
                            ("Amaun Elektrik", elec_amt),
                            ("kWh Before", kwh_b),
                            ("kWh After", kwh_a),
                            ("Total Units", total_u),
                        ]
                        elec_html = '<div class="doc-field-grid">'
                        for lbl, val in elec_pairs:
                            if val:
                                elec_html += f'<div class="doc-field"><div class="doc-field-label">{lbl}</div><div class="doc-field-value">{val}</div></div>'
                        elec_html += '</div>'
                        st.markdown(elec_html, unsafe_allow_html=True)

                    # ── Matching info ──────────────────────────────────
                    if matched_to:
                        conf = row.get("Match Confidence", "")
                        conf_icon = {"High": "🟢", "Medium": "🟡", "Low": "🟠"}.get(conf, "⚪")
                        st.markdown(
                            f'<div style="background:#f0fff4;border:1px solid #c6f6d5;border-radius:8px;padding:0.6rem 1rem;margin:0.4rem 0;">'
                            f'🔗 <strong>Matched To:</strong> {matched_to}<br/>'
                            f'{conf_icon} <strong>{conf}</strong> — {row.get("Matched On", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Additional details ─────────────────────────────
                    extras = [
                        ("Description", row.get("Description", "")),
                        ("Premise Address", row.get("Premise Address", "")),
                        ("No. Invois Cukai", row.get("No. Invois Cukai", "")),
                        ("Project", row.get("Project", "")),
                        ("Contract No / Batch No", row.get("Contract No / Batch No", "")),
                        ("LHDN UUID", row.get("LHDN UUID", "")),
                        ("Validate On", row.get("Validate On", "")),
                    ]
                    filled_extras = [(l, v) for l, v in extras if v]
                    if filled_extras:
                        ext_html = '<div class="doc-field-grid">'
                        for lbl, val in filled_extras:
                            ext_html += f'<div class="doc-field"><div class="doc-field-label">{lbl}</div><div class="doc-field-value">{val}</div></div>'
                        ext_html += '</div>'
                        st.markdown(ext_html, unsafe_allow_html=True)

                    src = row.get("_source_file", "")
                    if src:
                        st.caption(f"Source: {src}")

            st.markdown("---")

            # ── Table View ────────────────────────────────────────────
            st.markdown("#### 📊 Table View")
            display_cols = [c for c in REPORT_COLUMNS if c in filtered.columns]
            for extra in ["Matched To", "Match Confidence", "Matched On"]:
                if extra in filtered.columns:
                    display_cols.append(extra)
            table_df = filtered[display_cols].copy()
            table_df.insert(1, "Status", table_df["No"].apply(
                lambda n: st.session_state["doc_status"].get(int(n), "pending").capitalize()
            ))
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # ── Export ────────────────────────────────────────────────
            # Add status column to export
            display_cols = [c for c in REPORT_COLUMNS if c in filtered.columns]
            for extra in ["Matched To", "Match Confidence", "Matched On"]:
                if extra in filtered.columns:
                    display_cols.append(extra)
            export_df = filtered[display_cols].copy()
            export_df.insert(1, "Status", export_df["No"].apply(
                lambda n: st.session_state["doc_status"].get(int(n), "pending").capitalize()
            ))

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
                try:
                    import io
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        export_clean.to_excel(writer, index=False, sheet_name="Report")
                    st.download_button(
                        "⬇️ Download Excel",
                        data=excel_buffer.getvalue(),
                        file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except ImportError:
                    st.warning("Install `openpyxl` for Excel export: `pip install openpyxl`")

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
                                '<div style="background:#fff3cd; border-left:4px solid #ffc107; '
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
                                '<div style="background:#d4edda; border-left:4px solid #28a745; '
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
# PAGE: BANK MATCHING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🏦 Bank Matching":

    st.markdown("### 🏦 Bank Statement Matching & Reconciliation")
    st.markdown("Compare extracted invoices/bills against bank statement entries for automated reconciliation.")

    matching_json = Path(__file__).resolve().parent / "extraction_output" / "bank_matching_results.json"
    matching_txt = Path(__file__).resolve().parent / "extraction_output" / "bank_matching_report.txt"

    if matching_json.exists():
        mdata = load_json_file(matching_json)
        if isinstance(mdata, dict):
            display_bank_matching(mdata, matching_txt)
        else:
            st.error("Could not load matching results.")
    elif matching_txt.exists():
        st.markdown("#### 📄 Matching Report")
        st.code(matching_txt.read_text(encoding="utf-8"), language=None, line_numbers=True)
    else:
        st.info("No bank matching results found yet. Process a bank statement and documents first.")
