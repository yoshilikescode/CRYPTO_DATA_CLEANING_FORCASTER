"""
app.py
Professional Crypto Portfolio Dashboard
Bloomberg-inspired dark theme financial interface
"""
import os
import warnings
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from data_loader import load_crypto_data, load_live_crypto_data
from data_cleaner import clean_crypto_data
from index_builder import daily_index_builder, prepare_model_data
from factors_loader import load_factors
from forecasting_models import split_data, run_arima, run_sarimax

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="CryptoIndex Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────
# Bloomberg-inspired professional finance color palette
COLORS = {
    "bg_darkest":   "#010409",
    "bg_dark":      "#0d1117",
    "bg_surface":   "#161b22",
    "bg_elevated":  "#1c2128",
    "border":       "#21262d",
    "border_muted": "#161b22",
    "text_primary": "#e6edf3",
    "text_secondary":"#8b949e",
    "text_muted":   "#484f58",
    "accent_blue":  "#1f6feb",
    "accent_green": "#3fb950",
    "accent_red":   "#f85149",
    "accent_gold":  "#d29922",
    "accent_teal":  "#39d353",
    "chart_1":      "#1f6feb",
    "chart_2":      "#3fb950",
    "chart_3":      "#d29922",
    "chart_4":      "#f85149",
    "chart_5":      "#bc8cff",
    "chart_6":      "#39d353",
    "chart_7":      "#ff7b72",
    "chart_8":      "#79c0ff",
}

# ── PROFESSIONAL CSS ──────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── GLOBAL RESET ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLORS["bg_darkest"]} !important;
        color: {COLORS["text_primary"]} !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    .main {{
        background-color: {COLORS["bg_darkest"]} !important;
        padding: 0 !important;
    }}

    /* ── HIDE STREAMLIT BRANDING ── */
    /*#MainMenu, footer, header {{display: none !important;}}
    [data-testid="stToolbar"] {{display: none !important;}}
    .stDeployButton {{display: none !important;}}*/

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {{
        background-color: {COLORS["bg_dark"]} !important;
        border-right: 1px solid {COLORS["border"]} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: {COLORS["text_primary"]} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── METRIC CARDS ── */
    [data-testid="stMetric"] {{
        background-color: {COLORS["bg_surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
        padding: 16px 20px !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.8px !important;
        text-transform: uppercase !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {COLORS["text_primary"]} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }}

    /* ── BUTTONS ── */
    .stButton > button {{
        background-color: {COLORS["accent_blue"]} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        letter-spacing: 0.3px !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
    }}

    .stButton > button:hover {{
        background-color: #388bfd !important;
        transform: translateY(-1px) !important;
    }}

    /* ── INPUTS & SELECTS ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stSlider > div {{
        background-color: {COLORS["bg_surface"]} !important;
        border-color: {COLORS["border"]} !important;
        color: {COLORS["text_primary"]} !important;
        border-radius: 6px !important;
    }}

    /* ── RADIO BUTTONS ── */
    .stRadio > div {{
        background-color: {COLORS["bg_surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
        padding: 8px !important;
    }}

    /* ── EXPANDER ── */
    .streamlit-expanderHeader {{
        background-color: {COLORS["bg_surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
        color: {COLORS["text_secondary"]} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
    }}

    .streamlit-expanderContent {{
        background-color: {COLORS["bg_surface"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-top: none !important;
    }}

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
    }}

    /* ── DIVIDER ── */
    hr {{
        border-color: {COLORS["border"]} !important;
        margin: 24px 0 !important;
    }}

    /* ── SUCCESS/INFO/WARNING BOXES ── */
    .stSuccess {{
        background-color: rgba(35, 134, 54, 0.15) !important;
        border: 1px solid {COLORS["accent_green"]} !important;
        border-radius: 6px !important;
        color: {COLORS["accent_green"]} !important;
    }}

    .stInfo {{
        background-color: rgba(31, 111, 235, 0.15) !important;
        border: 1px solid {COLORS["accent_blue"]} !important;
        border-radius: 6px !important;
    }}

    .stWarning {{
        background-color: rgba(210, 153, 34, 0.15) !important;
        border: 1px solid {COLORS["accent_gold"]} !important;
        border-radius: 6px !important;
    }}

    /* ── SPINNER ── */
    .stSpinner > div {{
        border-top-color: {COLORS["accent_blue"]} !important;
    }}

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {COLORS["bg_surface"]} !important;
        border-bottom: 1px solid {COLORS["border"]} !important;
        gap: 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent !important;
        color: {COLORS["text_secondary"]} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        border-bottom: 2px solid transparent !important;
        padding: 8px 16px !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: {COLORS["text_primary"]} !important;
        border-bottom: 2px solid {COLORS["accent_blue"]} !important;
    }}

    /* ── SECTION HEADERS ── */
    .section-header {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {COLORS["text_secondary"]};
        padding: 0 0 12px 0;
        border-bottom: 1px solid {COLORS["border"]};
        margin-bottom: 16px;
    }}

    /* ── BADGE ── */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
    }}

    .badge-green {{
        background-color: rgba(63, 185, 80, 0.15);
        color: {COLORS["accent_green"]};
        border: 1px solid rgba(63, 185, 80, 0.3);
    }}

    .badge-blue {{
        background-color: rgba(31, 111, 235, 0.15);
        color: {COLORS["accent_blue"]};
        border: 1px solid rgba(31, 111, 235, 0.3);
    }}

    /* ── DOWNLOAD BUTTON ── */
    .stDownloadButton > button {{
        background-color: {COLORS["bg_surface"]} !important;
        color: {COLORS["text_secondary"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 6px !important;
        font-size: 12px !important;
    }}

    .stDownloadButton > button:hover {{
        border-color: {COLORS["accent_blue"]} !important;
        color: {COLORS["text_primary"]} !important;
    }}
</style>
""", unsafe_allow_html=True)


# ── PLOTLY THEME ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg_dark"],
    plot_bgcolor=COLORS["bg_surface"],
    font=dict(
        family="Inter, sans-serif",
        color=COLORS["text_secondary"],
        size=11
    ),
    xaxis=dict(
        gridcolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickcolor=COLORS["border"],
        tickfont=dict(
            family="JetBrains Mono",
            size=10,
            color=COLORS["text_muted"]
        ),
        showspikes=True,
        spikecolor=COLORS["border"],
        spikethickness=1,
        spikedash="dot"
    ),
    yaxis=dict(
        gridcolor=COLORS["border"],
        linecolor=COLORS["border"],
        tickcolor=COLORS["border"],
        tickfont=dict(
            family="JetBrains Mono",
            size=10,
            color=COLORS["text_muted"]
        ),
        showspikes=True,
        spikecolor=COLORS["border"],
        spikethickness=1,
        spikedash="dot"
    ),
    legend=dict(
        bgcolor=COLORS["bg_elevated"],
        bordercolor=COLORS["border"],
        borderwidth=1,
        font=dict(size=11, color=COLORS["text_secondary"])
    ),
    margin=dict(l=16, r=16, t=48, b=16),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=COLORS["bg_elevated"],
        bordercolor=COLORS["border"],
        font=dict(
            family="JetBrains Mono",
            size=11,
            color=COLORS["text_primary"]
        )
    )
)


# ── CHART FUNCTIONS ───────────────────────────────────────────
def plot_index_plotly(index_df):
    fig = go.Figure()

    # Area fill under line for depth
    fig.add_trace(go.Scatter(
        x=index_df.index,
        y=index_df["Index"],
        mode="lines",
        name="Portfolio Index",
        line=dict(
            color=COLORS["accent_blue"],
            width=1.5
        ),
        fill="tozeroy",
        fillcolor="rgba(31, 111, 235, 0.08)",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Index: %{y:.4f}<extra></extra>"
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="PORTFOLIO INDEX",
            font=dict(
                size=11,
                color=COLORS["text_secondary"],
                family="Inter",
            ),
            x=0.01
        ),
        height=380,
        showlegend=False
    )

    return fig


def plot_coins_plotly(cryptos_df, selected_coins):
    fig = go.Figure()
    # Force coin names to strings regardless of source
    selected_coins = [
        col if isinstance(col, str) else str(col)
        for col in selected_coins
    ]

    coin_colors = [
        COLORS["chart_1"], COLORS["chart_2"],
        COLORS["chart_3"], COLORS["chart_4"],
        COLORS["chart_5"], COLORS["chart_6"],
        COLORS["chart_7"], COLORS["chart_8"]
    ]

    for i, coin in enumerate(selected_coins):

        fig.add_trace(go.Scatter(
            x=cryptos_df.index,
            y=cryptos_df[coin],
            mode="lines",
            name=coin.upper(),
            line=dict(
                color=coin_colors[i % len(coin_colors)],
                width=1.5
            ),
            hovertemplate=f"<b>{coin.upper()}</b><br>%{{x|%b %d %Y}}<br>$%{{y:,.2f}}<extra></extra>"
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="ASSET PRICES  (USD)",
            font=dict(size=11, color=COLORS["text_secondary"]),
            x=0.01
        ),
        height=380
    )

    fig.update_yaxes(
        tickprefix="$",
        tickformat=",.0f"
    )


    return fig


def plot_forecast_plotly(y_test, prediction, ci, model_name):
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=ci.index.tolist() + ci.index.tolist()[::-1],
        y=ci.iloc[:, 0].tolist() + ci.iloc[:, 1].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(139, 148, 158, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence",
        hoverinfo="skip"
    ))

    # Actual line
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.iloc[:, 0],
        mode="lines",
        name="Actual",
        line=dict(color=COLORS["accent_blue"], width=1.5),
        hovertemplate="<b>Actual</b><br>%{x|%b %d %Y}: %{y:.4f}<extra></extra>"
    ))

    # Prediction line
    fig.add_trace(go.Scatter(
        x=prediction.index,
        y=prediction.values,
        mode="lines",
        name=f"{model_name}",
        line=dict(
            color=COLORS["accent_gold"],
            width=1.5,
            dash="dot"
        ),
        hovertemplate=f"<b>{model_name}</b><br>%{{x|%b %d %Y}}: %{{y:.4f}}<extra></extra>"
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{model_name} FORECAST  vs  ACTUAL",
            font=dict(size=11, color=COLORS["text_secondary"]),
            x=0.01
        ),
        height=380
    )

    return fig


def plot_model_comparison(y_test, predictions_dict):
    """Side by side model comparison chart"""
    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test.iloc[:, 0],
        mode="lines",
        name="Actual",
        line=dict(color=COLORS["accent_blue"], width=2),
        hovertemplate="<b>Actual</b>: %{y:.4f}<extra></extra>"
    ))

    colors = [COLORS["accent_gold"], COLORS["accent_green"],
              COLORS["accent_red"], COLORS["chart_5"]]

    for i, (name, data) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=data["pred"].index,
            y=data["pred"].values,
            mode="lines",
            name=name,
            line=dict(
                color=colors[i % len(colors)],
                width=1.5,
                dash="dot"
            ),
            hovertemplate=f"<b>{name}</b>: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="MODEL COMPARISON",
            font=dict(size=11, color=COLORS["text_secondary"]),
            x=0.01
        ),
        height=380
    )

    return fig


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # Logo / Title
    st.markdown(f"""
        <div style="padding: 8px 0 24px 0;">
            <div style="font-size:18px; font-weight:700;
                        color:{COLORS['text_primary']};
                        letter-spacing:-0.5px;">
                📈 CryptoIndex Predictor
            </div>
            <div style="font-size:11px; color:{COLORS['text_muted']};
                        margin-top:4px; letter-spacing:0.5px;">
                PORTFOLIO ANALYTICS
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">DATA SOURCE</div>',
                unsafe_allow_html=True)

    data_source = st.radio(
        "",
        ["Static  (2019 – 2022)", f"Live  (Yahoo Finance)"],
        label_visibility="collapsed"
    )

    if "Live" in data_source:
        st.markdown(f'<div class="section-header" style="margin-top:16px;">DATE RANGE</div>',
                    unsafe_allow_html=True)
        live_start = st.date_input("From", value=pd.to_datetime("2021-01-01"))
        live_end = st.date_input("To", value=date.today())
    else:
        live_start = pd.to_datetime("2021-01-01")
        live_end = date.today()

    st.markdown("---")
    st.markdown(f'<div class="section-header">MODELS</div>',
                unsafe_allow_html=True)

    models_to_run = st.multiselect(
        "",
        ["ARIMA", "SARIMAX"],
        default=["ARIMA"],
        label_visibility="collapsed"
    )

    st.markdown(f'<div class="section-header" style="margin-top:16px;">TRAIN / TEST SPLIT</div>',
                unsafe_allow_html=True)

    split = st.slider(
        "",
        min_value=80,
        max_value=98,
        value=95,
        format="%d%%",
        label_visibility="collapsed"
    ) / 100

    # Split preview
    st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
                    font-size:11px; color:{COLORS['text_muted']};
                    font-family:'JetBrains Mono'; margin-top:4px;">
            <span>TRAIN {int(split*100)}%</span>
            <span>TEST {int((1-split)*100)}%</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    run_button = st.button("▶  RUN FORECAST", use_container_width=True)

    st.markdown("---")

    # Footer
    st.markdown(f"""
        <div style="font-size:10px; color:{COLORS['text_muted']};
                    line-height:1.6;">
            <div>NOT FINANCIAL ADVICE</div>
            <div style="margin-top:4px;">Data: Yahoo Finance / Static CSV</div>
            <div>Models: ARIMA · SARIMAX</div>
        </div>
    """, unsafe_allow_html=True)


# ── MAIN LAYOUT ───────────────────────────────────────────────
# Header bar
st.markdown(f"""
    <div style="display:flex; align-items:center;
                justify-content:space-between;
                padding: 16px 0 8px 0;
                border-bottom: 1px solid {COLORS['border']};
                margin-bottom: 24px;">
        <div>
            <span style="font-size:20px; font-weight:700;
                         color:{COLORS['text_primary']};
                         letter-spacing:-0.5px;">
                Crypto Portfolio Dashboard
            </span>
            <span style="font-size:11px; color:{COLORS['text_muted']};
                         margin-left:12px; font-family:'JetBrains Mono';">
                EQUAL-WEIGHTED INDEX
            </span>
        </div>
        <div style="font-size:11px; color:{COLORS['text_muted']};
                    font-family:'JetBrains Mono';">
            {date.today().strftime("%b %d, %Y")}
        </div>
    </div>
""", unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
coins_path = os.path.join(
    base_dir, "CryptoCurrency", "data",
    "cryptocurrency", "coins", "*.csv"
)


@st.cache_data
def get_static_data():
    cryptos = load_crypto_data(coins_path, "2019-01-01", "2022-11-15")
    return clean_crypto_data(cryptos)


@st.cache_data
def get_live_data(start, end):
    cryptos = load_live_crypto_data(str(start), str(end))
    return clean_crypto_data(cryptos)


@st.cache_data
def get_factors(start_date, end_date):
    return load_factors(base_dir, start_date, end_date)


with st.spinner("Loading market data..."):
    if "Static" in data_source:
        cryptos_clean, start_date, end_date = get_static_data()
        source_badge = f'<span class="badge badge-blue">STATIC  2019 – 2022</span>'
    else:
        cryptos_clean, start_date, end_date = get_live_data(
            live_start, live_end
        )
        # Force all column names to clean strings
        # yfinance sometimes returns tuples as column names
        cryptos_clean.columns = [
            c if isinstance(c, str) else c[-1]
            for c in cryptos_clean.columns
        ]
        source_badge = f'<span class="badge badge-green">LIVE  {live_end}</span>'

# Safety net for both sources
cryptos_clean.columns = [
    c if isinstance(c, str) else str(c)
    for c in cryptos_clean.columns
]

# Build portfolio index from clean data
index = daily_index_builder(cryptos_clean)


# ── METRICS ROW ───────────────────────────────────────────────
st.markdown(f'<div class="section-header">PORTFOLIO OVERVIEW &nbsp; {source_badge}</div>',
            unsafe_allow_html=True)

current = index["Index"].iloc[-1]
start_val = index["Index"].iloc[0]
peak = index["Index"].max()
trough = index["Index"].min()
total_ret = ((current - start_val) / start_val) * 100
peak_drawdown = ((trough - peak) / peak) * 100

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "CURRENT INDEX",
    f"{current:.4f}",
    f"{total_ret:+.2f}%"
)
col2.metric(
    "PEAK VALUE",
    f"{peak:.4f}"
)
col3.metric(
    "TROUGH VALUE",
    f"{trough:.4f}"
)
col4.metric(
    "MAX DRAWDOWN",
    f"{peak_drawdown:.1f}%"
)
col5.metric(
    "DAYS TRACKED",
    f"{len(index):,}"
)

st.markdown("<br>", unsafe_allow_html=True)


# ── INDEX CHART ───────────────────────────────────────────────
st.markdown(f'<div class="section-header">INDEX PERFORMANCE</div>',
            unsafe_allow_html=True)
st.plotly_chart(
    plot_index_plotly(index),
    use_container_width=True
)


# ── COIN PRICES ───────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-header">ASSET BREAKDOWN</div>',
            unsafe_allow_html=True)
st.write("COLUMNS:")
st.write(cryptos_clean.columns)
selected_coins = st.multiselect(
    "",
    options=cryptos_clean.columns.tolist(),
    default=cryptos_clean.columns.tolist()[:4],
    label_visibility="collapsed"
)

if selected_coins:
    st.plotly_chart(
        plot_coins_plotly(cryptos_clean, selected_coins),
        use_container_width=True
    )


# ── RAW DATA INSPECTOR ────────────────────────────────────────
st.markdown("---")
with st.expander("RAW DATA INSPECTOR"):
    tab1, tab2 = st.tabs(["COIN PRICES", "INDEX VALUES"])

    with tab1:
        st.dataframe(
            cryptos_clean.style.highlight_max(
                axis=0, color="#1a472a"
            ).highlight_min(
                axis=0, color="#4a1c1c"
            ).format("{:.4f}"),
            use_container_width=True
        )

    with tab2:
        st.dataframe(
            index.style.highlight_max(
                axis=0, color="#1a472a"
            ).highlight_min(
                axis=0, color="#4a1c1c"
            ).format("{:.6f}"),
            use_container_width=True
        )


# ── DOWNLOAD ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-header">EXPORT</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])

col1.download_button(
    label="↓  Coin Prices CSV",
    data=cryptos_clean.to_csv().encode("utf-8"),
    file_name=f"coin_prices_{start_date.date()}_to_{end_date.date()}.csv",
    mime="text/csv",
    use_container_width=True
)

col2.download_button(
    label="↓  Index CSV",
    data=index.to_csv().encode("utf-8"),
    file_name=f"portfolio_index_{start_date.date()}_to_{end_date.date()}.csv",
    mime="text/csv",
    use_container_width=True
)


# ── FORECASTING ───────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div class="section-header">FORECASTING MODELS</div>',
            unsafe_allow_html=True)

if run_button:
    if not models_to_run:
        st.warning("Select at least one model from the sidebar.")
    else:
        with st.spinner("Preparing model data..."):
            factors = get_factors(start_date, end_date)
            data_prepared = prepare_model_data(index, factors)
            X_train, X_test, y_train, y_test = split_data(
                data_prepared, split
            )

        results = {}

        if "ARIMA" in models_to_run:
            with st.spinner("Training ARIMA..."):
                pred, rmse, ci = run_arima(y_train, y_test)
                results["ARIMA"] = {"pred": pred, "rmse": rmse, "ci": ci}
                st.plotly_chart(
                    plot_forecast_plotly(y_test, pred, ci, "ARIMA"),
                    use_container_width=True
                )

        if "SARIMAX" in models_to_run:
            with st.spinner("Training SARIMAX  (this takes a few minutes)..."):
                pred, rmse, ci = run_sarimax(
                    X_train, X_test, y_train, y_test
                )
                results["SARIMAX"] = {"pred": pred, "rmse": rmse, "ci": ci}
                st.plotly_chart(
                    plot_forecast_plotly(y_test, pred, ci, "SARIMAX"),
                    use_container_width=True
                )

        # Model comparison chart if both ran
        if len(results) > 1:
            st.markdown(f'<div class="section-header" style="margin-top:24px;">MODEL COMPARISON</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_model_comparison(y_test, results),
                use_container_width=True
            )

        # Results table
        st.markdown(f'<div class="section-header" style="margin-top:24px;">RESULTS SUMMARY</div>',
                    unsafe_allow_html=True)

        result_cols = st.columns(len(results))
        for i, (name, data) in enumerate(results.items()):
            result_cols[i].metric(
                f"{name}  RMSE",
                f"{data['rmse']:.6f}",
                "lower = better accuracy"
            )

else:
    st.markdown(f"""
        <div style="background-color:{COLORS['bg_surface']};
                    border: 1px solid {COLORS['border']};
                    border-radius:6px; padding:32px;
                    text-align:center;">
            <div style="font-size:13px; color:{COLORS['text_secondary']};">
                Configure settings in the sidebar and click
                <span style="color:{COLORS['accent_blue']};
                             font-weight:600;">▶ RUN FORECAST</span>
                to begin
            </div>
        </div>
    """, unsafe_allow_html=True)



