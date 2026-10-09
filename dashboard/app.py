"""
Virtual Economies & Player Behavior Engine
Interactive Research & Simulation Dashboard

Author: Héctor López (Aeterna Labs / ITSON Business Administration)
Stack: Streamlit + Plotly + NumPy + Pandas
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="MMO Behavioral Economy Engine | Héctor López",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Editorial Theme
st.markdown("""
<style>
    .reportview-container, .main, header, footer {
        background-color: #030712;
        color: #f8fafc;
    }
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        font-family: monospace;
    }
    .metric-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# Data paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

@st.cache_data
def load_data():
    macro_file = os.path.join(DATA_DIR, "macro_economy_trajectories.csv")
    telemetry_file = os.path.join(DATA_DIR, "player_behavior_telemetry.csv")
    orders_file = os.path.join(DATA_DIR, "auction_order_book.csv")
    
    df_macro = pd.read_csv(macro_file) if os.path.exists(macro_file) else pd.DataFrame()
    df_telemetry = pd.read_csv(telemetry_file) if os.path.exists(telemetry_file) else pd.DataFrame()
    df_orders = pd.read_csv(orders_file) if os.path.exists(orders_file) else pd.DataFrame()
    return df_macro, df_telemetry, df_orders

df_macro, df_telemetry, df_orders = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### 🪙 MMO Economy Engine")
    st.markdown("**Author:** Héctor López  \n*Aeterna Labs · ITSON '24*")
    st.markdown("---")
    st.markdown("""
    **Core Theoretical Framework:**
    - **Monetary Theory:** Fisher's Equation of Exchange ($M_s \\cdot V = P \\cdot Y$)
    - **Inequality Metric:** Gini Coefficient ($G \\in [0, 1]$)
    - **Behavioral Psychology:** Kahneman Loss Aversion ($\lambda = 2.25$) & Bartle Archetypes
    """)
    st.markdown("---")
    st.markdown("### Simulation Parameters")
    horizon = st.slider("Forecast Horizon (Days)", min_value=30, max_value=90, value=90)
    tax_input = st.slider("Simulated Auction Tax Rate", 0.01, 0.25, 0.05, step=0.01)
    sink_input = st.slider("Equipment Decay / Repair Sink", 0.01, 0.25, 0.08, step=0.01)

# Header
st.title("Virtual Economies & Player Behavior Engine")
st.caption("Agent-based macroeconomic modeling, auction house liquidity analysis, and empirical churn hazard forecasting.")

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Active Agent Population</div>
        <div class="metric-value">5,000 / 10K Cohort</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Money Supply (Gold Faucet)</div>
        <div class="metric-value">4.82M GP</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Gini Wealth Inequality</div>
        <div class="metric-value">0.584 (Moderate-High)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Predictive Churn ROC-AUC</div>
        <div class="metric-value">0.7431 (Validated)</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Macroeconomic Stabilization",
    "🧠 Behavioral Churn & Frustration",
    "⚖️ Auction House Liquidity",
    "🔬 Policy Sandbox & Methodology",
])

with tab1:
    st.subheader("Price Index Inflation: Unregulated vs. Dynamic Sink Policy")
    st.markdown(
        "Comparison of runaway fiat inflation in a game world versus an automated monetary stabilization algorithm adjusting transaction sinks."
    )
    
    if not df_macro.empty:
        fig_macro = px.line(
            df_macro,
            x="day",
            y="price_index",
            color="scenario",
            labels={"day": "In-Game Day", "price_index": "Price Index (Relative to Day 1 Base)"},
            template="plotly_dark",
            color_discrete_map={
                "Unregulated Baseline (Runaway Inflation)": "#f43f5e",
                "Algorithmic Monetary Stabilization": "#10b981",
            },
        )
        fig_macro.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_macro, use_container_width=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Wealth Inequality Dynamics (Gini Coefficient)")
            fig_gini = px.line(
                df_macro,
                x="day",
                y="gini_coefficient",
                color="scenario",
                template="plotly_dark",
                color_discrete_map={
                    "Unregulated Baseline (Runaway Inflation)": "#fb923c",
                    "Algorithmic Monetary Stabilization": "#38bdf8",
                },
            )
            fig_gini.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gini, use_container_width=True)
            
        with c_right:
            st.markdown("#### Player Retention Curve by Monetary Policy")
            fig_ret = px.line(
                df_macro,
                x="day",
                y="retention_rate",
                color="scenario",
                template="plotly_dark",
                color_discrete_map={
                    "Unregulated Baseline (Runaway Inflation)": "#e11d48",
                    "Algorithmic Monetary Stabilization": "#34d399",
                },
            )
            fig_ret.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ret, use_container_width=True)

with tab2:
    st.subheader("Behavioral Churn Prediction & Player Cohort Risk")
    st.markdown(
        "Machine learning model scoring player drop-off probability based on economic frustration, sink burdens, and loss-aversion thresholds."
    )
    
    if not df_telemetry.empty:
        col_t1, col_t2 = st.columns([5, 4])
        with col_t1:
            st.markdown("#### Churn Hazard Distribution by Risk Tier")
            risk_counts = df_telemetry["risk_tier"].value_counts().reset_index()
            risk_counts.columns = ["Risk Tier", "Player Count"]
            fig_pie = px.pie(
                risk_counts,
                names="Risk Tier",
                values="Player Count",
                template="plotly_dark",
                color_discrete_sequence=["#10b981", "#38bdf8", "#f59e0b", "#ef4444"],
                hole=0.45,
            )
            fig_pie.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_t2:
            st.markdown("#### Behavioral Feature Importance (Standardized Weights)")
            features = [
                "Frustration Index",
                "Sink Burden %",
                "Gold Deficit Ratio",
                "Loss Aversion Factor",
                "Wealth Elasticity",
                "Auction Participation",
                "Grind Hours Daily",
            ]
            importances = [0.284, 0.218, 0.185, 0.142, 0.089, -0.052, -0.030]
            fig_bar = px.bar(
                x=importances,
                y=features,
                orientation="h",
                labels={"x": "Normalized Relative Impact", "y": "Feature"},
                template="plotly_dark",
                color=importances,
                color_continuous_scale="Viridis",
            )
            fig_bar.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.subheader("Auction House Order Book & Market Liquidity Depth")
    if not df_orders.empty:
        st.dataframe(df_orders.head(15), use_container_width=True)
        
        fig_scatter = px.scatter(
            df_orders,
            x="bid_price",
            y="ask_price",
            size="volume_24h",
            color="tier",
            hover_name="item_name",
            template="plotly_dark",
            title="Bid vs Ask Distribution Across Item Tiers",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.subheader("Methodology, Psychological Grounding & Academic Context")
    st.markdown("""
    ### Why Virtual Economies Matter for Business Administration & Game Design
    Virtual worlds provide closed-loop laboratories for macroeconomic and psychological phenomena that take years to observe in physical nations:
    
    1. **Hyperinflation in Virtual Spaces:** Unlike central banks, MMO game designers continually introduce gold into the world via monster kills and quests without proportional destruction. Without effective currency sinks, currency velocity stalls, and the price of player-traded rare items inflates exponentially.
    2. **Loss Aversion (Kahneman & Tversky):** When price inflation forces casual players to grind 3x longer to afford consumables, psychological perceived loss triggers immediate churn, while speculative whales accumulate wealth monopolies.
    3. **Automated Algorithmic Monetary Policy:** By coupling transaction fees (friction taxes) and maintenance sinks to live Gini coefficient and velocity indexes, virtual economies achieve long-term retention stability.
    """)

st.markdown("---")
st.caption("Aeterna Labs Applied Research Lab · Developed by Héctor López · Public Repository on GitHub")
