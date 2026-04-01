"""
Shopee Seller Reconciliation — Interactive Dashboard
=====================================================
Streamlit UI that visualises:
  • Original data (Orders, Releases, Withdrawals, Bank Statements)
  • Mapping process (how records get matched)
  • Reconciliation results & summary metrics

Run:  streamlit run reconciliation_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import all logic from the existing demo module
from reconciliation_demo import (
    generate_sample_data,
    reconcile_orders,
    reconcile_withdrawals,
    _score_amount,
    _score_date,
    _score_description,
)

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Reconciliation Dashboard",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------
# Colour helpers for status badges
# ---------------------------------------------------------------
STATUS_COLORS = {
    "matched":           "#2ecc71",
    "timing_difference": "#f39c12",
    "amount_difference": "#e67e22",
    "missing_release":   "#e74c3c",
    "auto_match":        "#2ecc71",
    "suggested_match":   "#f39c12",
    "manual_review":     "#e74c3c",
}


def _colour_status(val):
    """Return inline CSS for status cells."""
    colour = STATUS_COLORS.get(val, "#95a5a6")
    return f"background-color: {colour}; color: white; border-radius: 4px; padding: 2px 8px;"


# ---------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
num_orders = st.sidebar.slider("Number of orders", 20, 200, 50, step=10)
num_withdrawals = st.sidebar.slider("Number of withdrawals", 3, 15, 5)

if st.sidebar.button("🔄 Regenerate Data", type="primary"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "1. Synthetic Shopee seller data is generated.\n"
    "2. *Order → Release* matching uses **key join + rules**.\n"
    "3. *Withdrawal → Bank* matching uses **weighted scoring + fuzzy matching**."
)

# ---------------------------------------------------------------
# Generate / cache data
# ---------------------------------------------------------------
@st.cache_data
def load_data(n_orders, n_wd):
    data = generate_sample_data(num_orders=n_orders, num_withdrawals=n_wd)
    order_recon = reconcile_orders(data["orders"], data["seller_balance_release"])
    wd_recon = reconcile_withdrawals(data["withdrawals"], data["bank_statement"])
    return data, order_recon, wd_recon


data, df_order_recon, df_wd_recon = load_data(num_orders, num_withdrawals)

df_orders = data["orders"]
df_releases = data["seller_balance_release"]
df_withdrawals = data["withdrawals"]
df_bank = data["bank_statement"]

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🔍 Shopee Seller Reconciliation Dashboard")
st.caption("Interactive demo — synthetic data with intentional anomalies")

# ===============================================================
# TAB LAYOUT
# ===============================================================
tab_summary, tab_data, tab_order, tab_wd = st.tabs([
    "📊 Summary",
    "📄 Original Data",
    "🔗 Order → Release Mapping",
    "🏦 Withdrawal → Bank Mapping",
])

# ---------------------------------------------------------------
# TAB 1 — SUMMARY
# ---------------------------------------------------------------
with tab_summary:
    st.header("Reconciliation Summary")

    col1, col2 = st.columns(2)

    # --- Order-level metrics ---
    with col1:
        st.subheader("Order → Seller Balance")
        o_counts = df_order_recon["status"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched",   o_counts.get("matched", 0))
        m2.metric("Timing Δ",  o_counts.get("timing_difference", 0))
        m3.metric("Amount Δ",  o_counts.get("amount_difference", 0))
        m4.metric("Missing",   o_counts.get("missing_release", 0))

        fig_o = px.pie(
            names=o_counts.index,
            values=o_counts.values,
            color=o_counts.index,
            color_discrete_map=STATUS_COLORS,
            hole=0.45,
        )
        fig_o.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_o, width="stretch")

    # --- Withdrawal-level metrics ---
    with col2:
        st.subheader("Withdrawal → Bank")
        w_counts = df_wd_recon["status"].value_counts()
        m5, m6, m7 = st.columns(3)
        m5.metric("Auto Match",     w_counts.get("auto_match", 0))
        m6.metric("Suggested",      w_counts.get("suggested_match", 0))
        m7.metric("Manual Review",  w_counts.get("manual_review", 0))

        fig_w = px.pie(
            names=w_counts.index,
            values=w_counts.values,
            color=w_counts.index,
            color_discrete_map=STATUS_COLORS,
            hole=0.45,
        )
        fig_w.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_w, width="stretch")

    # --- Overall health bar ---
    st.markdown("---")
    total = len(df_order_recon)
    matched_pct = round(o_counts.get("matched", 0) / total * 100, 1) if total else 0
    st.subheader(f"Overall Order Match Rate: {matched_pct}%")
    st.progress(matched_pct / 100)


# ---------------------------------------------------------------
# TAB 2 — ORIGINAL DATA
# ---------------------------------------------------------------
with tab_data:
    st.header("Original Data")

    d1, d2 = st.columns(2)
    with d1:
        st.subheader(f"📦 Orders ({len(df_orders)} rows)")
        st.dataframe(df_orders, width="stretch", height=350)

    with d2:
        st.subheader(f"💰 Seller Balance Releases ({len(df_releases)} rows)")
        st.dataframe(df_releases, width="stretch", height=350)

    d3, d4 = st.columns(2)
    with d3:
        st.subheader(f"🏧 Withdrawals ({len(df_withdrawals)} rows)")
        st.dataframe(df_withdrawals, width="stretch", height=250)

    with d4:
        st.subheader(f"🏦 Bank Statement ({len(df_bank)} rows)")
        st.dataframe(df_bank, width="stretch", height=250)


# ---------------------------------------------------------------
# TAB 3 — ORDER → RELEASE MAPPING
# ---------------------------------------------------------------
with tab_order:
    st.header("Order → Seller Balance Release Mapping")

    # --- Filter controls ---
    status_filter = st.multiselect(
        "Filter by status",
        options=df_order_recon["status"].unique().tolist(),
        default=df_order_recon["status"].unique().tolist(),
        key="order_status_filter",
    )
    filtered = df_order_recon[df_order_recon["status"].isin(status_filter)]

    # --- Styled table ---
    st.subheader(f"Results ({len(filtered)} records)")
    st.dataframe(
        filtered.style.map(_colour_status, subset=["status"]),
        width="stretch",
        height=400,
    )

    # --- Visual: mapping flow (Sankey-style) ---
    st.subheader("Mapping Flow")
    st.caption("Each order flows to a reconciliation status category")

    status_list = ["matched", "timing_difference", "amount_difference", "missing_release"]
    counts_for_sankey = [int(o_counts.get(s, 0)) for s in status_list]

    labels = ["Orders"] + [s.replace("_", " ").title() for s in status_list]
    source = [0] * len(status_list)
    target = list(range(1, len(status_list) + 1))
    values = counts_for_sankey
    colours_link = [STATUS_COLORS.get(s, "#95a5a6") for s in status_list]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=25,
            label=labels,
            color=["#3498db"] + [STATUS_COLORS.get(s, "#95a5a6") for s in status_list],
        ),
        link=dict(source=source, target=target, value=values, color=colours_link),
    ))
    fig_sankey.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sankey, width="stretch")

    # --- Date-difference histogram ---
    df_with_dates = filtered.dropna(subset=["date_difference"])
    if not df_with_dates.empty:
        st.subheader("Release Delay Distribution (days)")
        fig_hist = px.histogram(
            df_with_dates, x="date_difference", nbins=10,
            color="status", color_discrete_map=STATUS_COLORS,
            labels={"date_difference": "Days between completion & release"},
        )
        fig_hist.update_layout(height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, width="stretch")

    # --- Amount scatter ---
    df_with_amt = filtered.dropna(subset=["amount_difference_value"])
    if not df_with_amt.empty:
        st.subheader("Amount Comparison")
        fig_scatter = px.scatter(
            df_with_amt,
            x="net_amount",
            y="release_amount",
            color="status",
            color_discrete_map=STATUS_COLORS,
            hover_data=["order_id", "amount_difference_value"],
            labels={"net_amount": "Expected (net_amount)", "release_amount": "Actual (release_amount)"},
        )
        # Add perfect-match diagonal line
        max_val = max(df_with_amt["net_amount"].max(), df_with_amt["release_amount"].max()) * 1.05
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(dash="dash", color="grey"),
        )
        fig_scatter.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_scatter, width="stretch")


# ---------------------------------------------------------------
# TAB 4 — WITHDRAWAL → BANK MAPPING
# ---------------------------------------------------------------
with tab_wd:
    st.header("Withdrawal → Bank Statement Mapping")

    # --- Result table ---
    st.subheader("Match Results")
    st.dataframe(
        df_wd_recon.style.map(_colour_status, subset=["status"]),
        width="stretch",
    )

    # --- Score breakdown detail ---
    st.subheader("Score Breakdown per Withdrawal")
    st.caption("Shows how each score component contributes to the total match score")

    breakdown_rows = []
    for _, wd in df_withdrawals.iterrows():
        wd_id = wd["withdrawal_id"]
        wd_date = wd["withdrawal_date"]
        wd_net = round(wd["withdrawal_amount"] - wd["withdrawal_fee"], 2)

        for _, bk in df_bank.iterrows():
            amt_score = _score_amount(wd_net, bk["bank_amount"])
            date_score = _score_date(wd_date, bk["bank_date"])
            desc_score = _score_description(bk["description"])
            total = amt_score + date_score + desc_score
            breakdown_rows.append({
                "withdrawal_id": wd_id,
                "bank_description": bk["description"],
                "bank_amount": bk["bank_amount"],
                "amount_score (50)": amt_score,
                "date_score (30)": date_score,
                "desc_score (20)": desc_score,
                "total_score (100)": total,
            })

    df_breakdown = pd.DataFrame(breakdown_rows)

    # Let user pick a withdrawal to inspect
    wd_pick = st.selectbox(
        "Select withdrawal to inspect",
        df_withdrawals["withdrawal_id"].tolist(),
    )
    df_pick = df_breakdown[df_breakdown["withdrawal_id"] == wd_pick].sort_values(
        "total_score (100)", ascending=False
    )
    st.dataframe(df_pick, width="stretch")

    # Stacked bar chart of score components for the selected withdrawal
    if not df_pick.empty:
        fig_bar = go.Figure()
        for col, colour in [
            ("amount_score (50)", "#2ecc71"),
            ("date_score (30)", "#3498db"),
            ("desc_score (20)", "#9b59b6"),
        ]:
            fig_bar.add_trace(go.Bar(
                x=df_pick["bank_description"],
                y=df_pick[col],
                name=col,
                marker_color=colour,
            ))
        fig_bar.update_layout(
            barmode="stack",
            yaxis_title="Score",
            xaxis_title="Bank Entry",
            height=320,
            margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.12),
        )
        # Threshold lines
        fig_bar.add_hline(y=90, line_dash="dash", line_color="#2ecc71",
                          annotation_text="Auto-match (90)")
        fig_bar.add_hline(y=70, line_dash="dot", line_color="#f39c12",
                          annotation_text="Suggested (70)")
        st.plotly_chart(fig_bar, width="stretch")

    # --- Mapping flow (Sankey) ---
    st.subheader("Withdrawal Mapping Flow")

    wd_labels = df_wd_recon["withdrawal_id"].tolist()
    status_labels = sorted(df_wd_recon["status"].unique().tolist())
    all_labels = wd_labels + status_labels

    san_source, san_target, san_value, san_colour = [], [], [], []
    for i, row in df_wd_recon.iterrows():
        src_idx = wd_labels.index(row["withdrawal_id"])
        tgt_idx = len(wd_labels) + status_labels.index(row["status"])
        san_source.append(src_idx)
        san_target.append(tgt_idx)
        san_value.append(1)
        san_colour.append(STATUS_COLORS.get(row["status"], "#95a5a6"))

    fig_san2 = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            label=all_labels,
            color=(
                ["#3498db"] * len(wd_labels)
                + [STATUS_COLORS.get(s, "#95a5a6") for s in status_labels]
            ),
        ),
        link=dict(source=san_source, target=san_target, value=san_value, color=san_colour),
    ))
    fig_san2.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_san2, width="stretch")
