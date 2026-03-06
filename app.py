import os
import streamlit as st              # type: ignore
import pandas as pd                 # type: ignore
import matplotlib.pyplot as plt     # type: ignore
import seaborn as sns               # type: ignore
import numpy as np                  # type: ignore
from pathlib import Path
from scipy import stats             # type: ignore
import plotly.express as px         # type: ignore
import plotly.graph_objects as go   # type: ignore
from scipy.stats import pearsonr    # type: ignore
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("merged_companies_housing.csv")
    year_series = pd.to_datetime(df['date'], errors='coerce').dt.year
    if year_series.isna().mean() > 0.5:
        year_series = df['date'].astype(str).str.extract(r'(\b\d{4}\b)')[0].astype(float)
    if 'year' in df.columns and df['year'].notna().mean() > 0.5:
        year_series = pd.to_numeric(df['year'], errors='coerce')
    df['year_int'] = year_series.astype('Int64')
    df = df.dropna(subset=['year_int'])
    df['year_int'] = df['year_int'].astype(int)
    df['city_x'] = df['city_x'].fillna("Unknown").astype(str)
    df = df.dropna(subset=['pct_change'])
    return df

@st.cache_data
def load_integrated_data():
    possible_names = [
        "firmscape_integrated_cbsa_quarterly_cleaned.csv",
        "firmscape_integrated_cbsa.csv",
        "firmscape_integrated_cbsa_quarterly.csv",
        "firmscape_active_businesses.csv",
        "firmscape_active_businesse.csv",
    ]
    import glob
    found = glob.glob("firmscape_integrated*.csv") + glob.glob("firmscape_active*.csv")
    all_names = possible_names + [f for f in found if f not in possible_names]

    for fname in all_names:
        try:
            df = pd.read_csv(fname)
            for city_candidate in ['city_state', 'city', 'metro_name', 'cbsa_name']:
                if city_candidate in df.columns:
                    df[city_candidate] = df[city_candidate].fillna("Unknown").astype(str)
                    df['_city_col'] = city_candidate
                    break
            df['year'] = pd.to_numeric(df.get('year', pd.Series(dtype=float)), errors='coerce')
            df['quarter'] = pd.to_numeric(df.get('quarter', pd.Series(dtype=float)), errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            if 'yq' not in df.columns:
                df['yq'] = df['year'].astype(str) + "Q" + df['quarter'].fillna(1).astype(int).astype(str)
            else:
                df['yq'] = df['yq'].astype(str)
            for col in ['fhfa_index', 'fhfa_yoy', 'fhfa_qoq', 'zillow_price_q',
                        'zillow_yoy', 'zillow_qoq', 'firms_founded_yoy',
                        'hhi_new', 'top_industry_share_new', 'industry_count_new',
                        'firm_count_total']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df, fname
        except FileNotFoundError:
            continue
    return None, None

try:
    merged_companies_housing = load_data()
except FileNotFoundError:
    st.error("File 'merged_companies_housing.csv' not found.")
    st.stop()

integrated_df, integrated_fname = load_integrated_data()

# ─────────────────────────────────────────────
# COMPUTE TOP CITIES FROM INTEGRATED DATASET
# ─────────────────────────────────────────────
def get_top_cities(idf, n=100):
    """Return top-n cities by data completeness from the integrated dataset."""
    if idf is None:
        return []
    city_col_name = idf['_city_col'].iloc[0] if '_city_col' in idf.columns else 'city_state'
    idf2 = idf.copy()
    idf2['city_state'] = idf2[city_col_name].astype(str)
    required_cols = ['fhfa_yoy', 'zillow_yoy', 'zillow_price_q', 'firms_founded_yoy']
    available_cols = [c for c in required_cols if c in idf2.columns]
    city_completeness = (
        idf2.groupby('city_state')[available_cols]
        .count()
        .min(axis=1)
        .sort_values(ascending=False)
    )
    return city_completeness[city_completeness > 0].head(n).index.tolist()

TOP_100_CITIES = get_top_cities(integrated_df, 100)

def resolve_city_from_keyword(keyword, city_list):
    """Find best-matching city from list for a given keyword."""
    return next((c for c in city_list if keyword.lower() in c.lower()), None)

# ─────────────────────────────────────────────
# CURATED CITIES — always exactly 5 valid cities
# Prefers iconic case-study cities by keyword;
# pads with top-data cities when keywords don’t match.
# ─────────────────────────────────────────────
CASE_STUDY_KEYWORDS = ["Detroit", "Austin", "Seattle", "Chicago", "Phoenix"]

# Build keyword → resolved dataset city map
KEYWORD_TO_CITY = {kw: resolve_city_from_keyword(kw, TOP_100_CITIES) for kw in CASE_STUDY_KEYWORDS}

# Start with matched keywords (preserving order)
CURATED_CITIES = [KEYWORD_TO_CITY[kw] for kw in CASE_STUDY_KEYWORDS if KEYWORD_TO_CITY[kw]]

# Pad to exactly 5 with next best top-data cities not already included
for city in TOP_100_CITIES:
    if len(CURATED_CITIES) >= 5:
        break
    if city not in CURATED_CITIES:
        CURATED_CITIES.append(city)

# Ultimate fallback
if not CURATED_CITIES:
    CURATED_CITIES = TOP_100_CITIES[:5]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("FirmScape Dashboard")

st.sidebar.markdown("### 👤 Who Are You?")
stakeholder = st.sidebar.radio(
    "Select your perspective:",
    ["🏠 Housing Investor", "📊 Business Analyst", "🔬 Researcher"]
)

tab = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🧩 Build the Dataset", "📊 EDA Explorer", "🔎 Evidence", "✅ Validation & Modeling", "🚀 Opportunity Lab"]
)

# ─────────────────────────────────────────────
# CASE STUDY PRESETS — fully data-driven
# Built from the actual top 5 cities with the
# most complete data in the integrated dataset.
# ─────────────────────────────────────────────
def build_case_study_presets(idf, curated_cities):
    """
    Generate case study presets for the top 5 curated cities,
    using real stats from the dataset to write honest narratives.
    """
    if idf is None or not curated_cities:
        return {}

    city_col_name = idf['_city_col'].iloc[0] if '_city_col' in idf.columns else 'city_state'
    idf2 = idf.copy()
    idf2['city_state'] = idf2[city_col_name].astype(str)

    EMOJIS = ["🏙️", "📈", "🏗️", "🌆", "🌇"]
    presets = {}

    for i, city in enumerate(curated_cities[:5]):
        cd = idf2[idf2['city_state'] == city]

        # — Housing stats —
        hvals = cd['fhfa_yoy'].dropna() if 'fhfa_yoy' in cd.columns else pd.Series(dtype=float)
        avg_hpi   = hvals.mean() * 100 if len(hvals) > 0 else None
        vol_hpi   = hvals.std()  * 100 if len(hvals) > 0 else None
        pct_pos   = (hvals > 0).mean() * 100 if len(hvals) > 0 else None
        max_yr    = int(cd.loc[cd['fhfa_yoy'].idxmax(), 'year']) if (len(hvals) > 0 and 'year' in cd.columns) else None
        min_yr    = int(cd.loc[cd['fhfa_yoy'].idxmin(), 'year']) if (len(hvals) > 0 and 'year' in cd.columns) else None

        # — Firm stats —
        fvals = cd['firms_founded_yoy'].dropna() if 'firms_founded_yoy' in cd.columns else pd.Series(dtype=float)
        avg_firm  = fvals.mean() * 100 if len(fvals) > 0 else None

        # — Diversity stats —
        dvals = cd['industry_count_new'].dropna() if 'industry_count_new' in cd.columns else pd.Series(dtype=float)
        avg_div   = int(dvals.mean()) if len(dvals) > 0 else None

        # — Build narrative from real numbers —
        short_name = city.split(",")[0].split("-")[0].strip()

        if avg_hpi is not None and vol_hpi is not None:
            if vol_hpi > 8:
                volatility_label = "highly volatile"
                vol_insight = f"with a std dev of {vol_hpi:.1f}% — one of the more turbulent markets in the dataset"
            elif vol_hpi > 4:
                volatility_label = "moderately volatile"
                vol_insight = f"with a std dev of {vol_hpi:.1f}% — typical boom-bust sensitivity"
            else:
                volatility_label = "relatively stable"
                vol_insight = f"with a std dev of {vol_hpi:.1f}% — more insulated from national shocks"

            peak_note = f"Peak growth occurred around {max_yr}." if max_yr else ""
            crash_note = f"The sharpest decline was around {min_yr}." if min_yr else ""

            story = (
                f"{short_name} averaged {avg_hpi:.1f}% annual home price growth and is {volatility_label} "
                f"({vol_insight}). Prices rose in {pct_pos:.0f}% of all quarters. "
                f"{peak_note} {crash_note}".strip()
            )
            if avg_firm is not None:
                firm_dir = "above" if avg_firm > 0 else "below"
                story += f" Firm founding averaged {abs(avg_firm):.1f}% per year — {firm_dir} the zero baseline."
            if avg_div is not None:
                story += f" The economy averaged {avg_div} distinct industries per quarter."

            what_to_look_for = (
                f"Watch the {('peaks around ' + str(max_yr)) if max_yr else 'highs'} "
                f"and {('troughs around ' + str(min_yr)) if min_yr else 'lows'} in the housing line. "
                f"Does firm growth lead or lag the housing moves?"
            )
        else:
            story = f"Explore {short_name}'s housing and firm founding data across 50 years of quarterly records."
            what_to_look_for = "Look for periods where firm growth accelerates before housing prices follow."

        presets[f"{EMOJIS[i]} {short_name}"] = {
            "city_keyword": short_name,
            "city_full": city,          # exact dataset name — used for direct lookup
            "housing_metric": "fhfa_yoy (FHFA % change YoY)",
            "story": story,
            "what_to_look_for": what_to_look_for,
            "emoji": EMOJIS[i],
        }

    return presets

CASE_STUDY_PRESETS = build_case_study_presets(integrated_df, CURATED_CITIES)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if 'case_study_preset' not in st.session_state:
    st.session_state['case_study_preset'] = None
if 'preset_city' not in st.session_state:
    st.session_state['preset_city'] = None
if 'preset_housing_metric' not in st.session_state:
    st.session_state['preset_housing_metric'] = None

# ─────────────────────────────────────────────
# HOME TAB
# ─────────────────────────────────────────────
if tab == "🏠 Home":
    st.title("FirmScape")

    # Research question — always shown
    st.markdown("""
    <div style="background:#1a1f2e; border-left:4px solid #4f8ef7; border-radius:8px; padding:14px 20px; margin-bottom:16px;">
        <p style="color:#aab4c8; font-size:1.05em; margin:0;">
            <strong style="color:white;">Research Question:</strong>
            <em> When industries cluster and grow in a city, how do housing prices move — now and later?</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    c_left, c_right = st.columns([2, 1])

    if stakeholder == "🏠 Housing Investor":
        with c_left:
            st.markdown("##### *Spot cities about to heat up — before the market prices it in.*")
            st.markdown("""
            Industry growth is a **leading indicator** for housing demand — typically 1–2 years ahead.

            **Start here:** → Evidence → Opportunity Lab
            """)
            with st.expander("⚠️ Scope & limits"):
                st.markdown("This is a **screening tool**, not a price forecast. Always layer in macro factors (rates, supply, zoning) before acting.")
        with c_right:
            st.metric("Cities / metros", "13,330")
            st.metric("Years of data", "50")
            st.metric("Avg lag signal", "1–2 yrs")
            st.metric("Data coverage", "96%")

    elif stakeholder == "📊 Business Analyst":
        with c_left:
            st.markdown("##### *Quantify how industry structure drives housing price changes.*")
            st.markdown("""
            Test which variables — HHI, firm founding rate, diversity — are statistically predictive and at what lag.

            **Start here:** → EDA Explorer → Validation & Modeling
            """)
            with st.expander("⚠️ Scope & limits"):
                st.markdown("A **low R²** is expected — industry structure is one signal among many. Goal: identify which variables matter most, not build a complete model.")
        with c_right:
            st.metric("Predictive variables", "4")
            st.metric("Models available", "3")
            st.metric("Lag options", "0–3 yrs")
            st.metric("Data coverage", "96%")

    else:  # Researcher
        with c_left:
            st.markdown("##### *50 years of quarterly panel data across 13,330 CBSAs.*")
            st.markdown("""
            All R², p-values, and lag structures shown transparently. Inspect data pipeline and methodology.

            **Start here:** → Build the Dataset → Validation & Modeling
            """)
            with st.expander("⚠️ Methodological notes"):
                st.markdown("""
                - All correlations are **associative** — no causal identification
                - Panel is **unbalanced** — smaller metros have fewer observations
                - Firm data reflects **registration**, not operational firms (survivorship bias applies)
                """)
        with c_right:
            st.metric("CBSAs covered", "13,330")
            st.metric("Quarterly obs.", "275,438")
            st.metric("Years of panel", "50")
            st.metric("Data coverage", "96%")

    st.divider()

    st.subheader("🗺️ Explore a Case Study")
    cs_names = list(CASE_STUDY_PRESETS.keys())
    selected_cs = st.selectbox("Pick a city:", cs_names, index=0, key="home_case_study")
    preset = CASE_STUDY_PRESETS[selected_cs]
    city_full = preset.get("city_full", CURATED_CITIES[0])

    st.markdown(f"""
    <div style="background: #1a1f2e; border-left: 4px solid #4f8ef7; border-radius: 8px; padding: 20px; margin: 12px 0;">
        <h3 style="margin:0; color:white;">{preset["emoji"]} {selected_cs}</h3>
        <p style="color: #aab4c8; margin: 10px 0 4px 0; font-size:0.85em;">📍 Dataset city: <strong style="color:#4f8ef7">{city_full}</strong></p>
        <p style="color: #aab4c8; margin: 6px 0 6px 0;">{preset["story"]}</p>
        <p style="color: #f7a44f; font-size: 0.88em; margin:0;">
            📌 <strong>What to look for:</strong> {preset["what_to_look_for"]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"🚀 Load {selected_cs} → jump to Evidence tab", type="primary"):
        st.session_state["case_study_preset"] = selected_cs
        st.session_state["preset_city"] = city_full
        st.session_state["preset_housing_metric"] = preset["housing_metric"]
        st.success(f"✅ Preset loaded! Go to **🔎 Evidence** — pre-set to **{city_full}**.")

    if st.session_state.get("case_study_preset"):
        active = st.session_state["case_study_preset"]
        active_city = st.session_state.get("preset_city", CURATED_CITIES[0])
        st.info(f"📍 Active preset: **{active}** → {active_city}. Go to **🔎 Evidence** to explore.")

    st.divider()

    st.subheader("How to use FirmScape")
    hw1, hw2, hw3, hw4 = st.columns(4)

    if stakeholder == "🏠 Housing Investor":
        with hw1:
            st.markdown("**1️⃣ EDA Explorer**")
            st.caption("Learn what each variable measures.")
        with hw2:
            st.markdown("**2️⃣ Evidence**")
            st.caption("See if firm growth leads housing in your target city.")
        with hw3:
            st.markdown("**3️⃣ Validation**")
            st.caption("Check which signals are statistically significant.")
        with hw4:
            st.markdown("**4️⃣ Opportunity Lab ⭐**")
            st.caption("Build your investment shortlist.")
    elif stakeholder == "📊 Business Analyst":
        with hw1:
            st.markdown("**1️⃣ EDA Explorer ⭐**")
            st.caption("Variable definitions, distributions, city comparisons.")
        with hw2:
            st.markdown("**2️⃣ Evidence**")
            st.caption("Scatter plots and multi-city trend comparisons.")
        with hw3:
            st.markdown("**3️⃣ Validation ⭐**")
            st.caption("Run models, compare R², test lags.")
        with hw4:
            st.markdown("**4️⃣ Opportunity Lab**")
            st.caption("Sensitivity analysis with custom weights.")
    else:
        with hw1:
            st.markdown("**1️⃣ Build the Dataset ⭐**")
            st.caption("Full pipeline: sources, cleaning, joins.")
        with hw2:
            st.markdown("**2️⃣ EDA Explorer**")
            st.caption("Variable properties and data quality.")
        with hw3:
            st.markdown("**3️⃣ Validation ⭐**")
            st.caption("p-values, R², lag structures — full output.")
        with hw4:
            st.markdown("**4️⃣ Evidence**")
            st.caption("Inspect city-level time series.")

    st.caption("*Disclaimer: No causal claims — all findings are associative and data-driven.*")

if tab == "🧩 Build the Dataset":
    st.title("From Messy Sources to One City Timeline")

    st.markdown("""
    **📊 Data Pipeline:**  
    Company data → Cleaned → Aggregated by city & year  
    Housing data → Cleaned → Aligned  
    Join → Integrated panel dataset
    """)

    pipeline_data = pd.DataFrame({
        "Dataset": ["companies_sorted.csv", "companies_sorted.csv",
                    "companies-2023-q4-sm.csv", "companies-2023-q4-sm.csv",
                    "hpi_at_metro.csv", "hpi_at_metro.csv",
                    "Zillow_Housing_Dataset.csv", "Zillow_Housing_Dataset.csv",
                    "Final Panel"],
        "Stage": ["Raw", "Cleaned", "Raw", "Cleaned",
                  "Raw", "Cleaned", "Raw", "Cleaned",
                  "Integrated"],
        "Rows": [7173426, 40258, 19486334, 0, 83230, 69828, 895, 230406, 275438],
        "Columns": [11, 7, 11, 0, 6, 6, 317, 7, 15],
        "Notes": [
            "Some missing values",
            "Filtered to US companies with 100+ employees; dropped unwanted columns",
            "Very large dataset",
            "Filtered to US companies with 100+ employees; dropped unwanted columns",
            "Needs to be recomputed for percent change",
            "Split location into city/state; created datetime column; dropped rows without HPI",
            "Many empty rows",
            "Sorted for time; filled small gaps; dropped NaNs",
            "Cleaned & Merged!"
        ]
    })
    st.dataframe(pipeline_data, height=500, width=900)

# ─────────────────────────────────────────────
# EDA EXPLORER TAB
# ─────────────────────────────────────────────
if tab == "📊 EDA Explorer":
    st.title("📊 EDA Explorer: What Are We Measuring?")
    st.markdown(
        "Select a variable below to see its definition, chart, and key stats. "
        "Expand the details sections for deeper context."
    )

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_var = st.selectbox("🔍 Select a variable:", [
            "🎯 Target: Housing Price Change",
            "📡 Signal 1: Firm Founding Rate",
            "📡 Signal 2: Industry Concentration (HHI)",
            "📡 Signal 3: Industry Diversity",
            "📡 Signal 4: Top Industry Share",
        ], key="eda_var")

    use_idf = integrated_df is not None
    if use_idf:
        idf_eda = integrated_df.copy()
        city_col_eda = idf_eda['_city_col'].iloc[0] if '_city_col' in idf_eda.columns else 'city_state'
        idf_eda['city_state'] = idf_eda[city_col_eda].astype(str)
        # Use top cities from dataset — prefer curated 5, pad with more top cities
        eda_cities = [c for c in CURATED_CITIES if c in TOP_100_CITIES] or TOP_100_CITIES[:5]
    else:
        idf_eda = None
        eda_cities = CURATED_CITIES

    with sel_col2:
        if "Target" in selected_var or "Signal 1" in selected_var:
            eda_city_pick = st.selectbox("Pick a city:", eda_cities, key="eda_city_global")
            eda_compare_cities = None
        elif "Signal 2" in selected_var:
            eda_compare_cities = st.multiselect("Compare cities:", eda_cities, default=eda_cities[:3], key="eda_hhi_cities")
            eda_city_pick = None
        elif "Signal 3" in selected_var:
            eda_compare_cities = st.multiselect("Compare cities:", eda_cities, default=eda_cities[:3], key="eda_div_cities")
            eda_city_pick = None
        elif "Signal 4" in selected_var:
            eda_compare_cities = st.multiselect("Compare cities:", eda_cities, default=eda_cities[:3], key="eda_top_cities")
            eda_city_pick = None
        else:
            eda_city_pick = None
            eda_compare_cities = None

    st.divider()

    def dark_fig(w=10, h=4):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        return fig, ax

    def chart_note(text):
        st.markdown(
            f'<p style="color:#888; font-size:0.8em; margin-top:4px;">ℹ️ {text}</p>',
            unsafe_allow_html=True
        )

    if "Target" in selected_var:
        st.markdown("#### 🎯 Housing Price Change (FHFA YoY %)")
        st.info("**What it is:** Year-over-year % change in average home prices — the variable we're trying to predict.")

        if use_idf:
            city_pick = eda_city_pick
            city_data = idf_eda[idf_eda['city_state'] == city_pick].dropna(subset=['fhfa_yoy']).sort_values(['year', 'quarter'])

            if not city_data.empty:
                fig, ax = dark_fig()
                ax.plot(range(len(city_data)), city_data['fhfa_yoy'].values, color='#4f8ef7', linewidth=2)
                ax.axhline(0, color='#666', linewidth=0.8, linestyle='--')
                ax.fill_between(range(len(city_data)), city_data['fhfa_yoy'].values, 0,
                                where=city_data['fhfa_yoy'].values > 0, alpha=0.2, color='#4f8ef7')
                ax.fill_between(range(len(city_data)), city_data['fhfa_yoy'].values, 0,
                                where=city_data['fhfa_yoy'].values < 0, alpha=0.2, color='#f74f4f')
                yr_rows = city_data.reset_index(drop=True)
                ticks = yr_rows[yr_rows['quarter'] == 1].index[::4].tolist()
                labels = yr_rows.loc[ticks, 'year'].astype(str).tolist()
                ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, color='white', fontsize=8)
                ax.set_ylabel("% change vs year before", color='white')
                ax.set_title(f"{city_pick} — Home Price Change Per Year (FHFA)", color='white')
                st.pyplot(fig); plt.close(fig)
                chart_note("Blue = prices rising · Red = prices falling · 2008 crash visible in most cities")

                hpi_vals = city_data['fhfa_yoy'].dropna()
                avg = hpi_vals.mean()
                volatility = hpi_vals.std()
                pct_positive = (hpi_vals > 0).mean() * 100
                col_h1, col_h2, col_h3 = st.columns(3)
                col_h1.metric("Avg annual price change", f"{avg*100:.1f}%")
                col_h2.metric("Volatility (std dev)", f"{volatility*100:.1f}%")
                col_h3.metric("Years with rising prices", f"{pct_positive:.0f}%")

                takeaway_text = (
                    f"Prices rose <b>{pct_positive:.0f}%</b> of the time, avg <b>{avg*100:.1f}%/yr</b>. "
                    f"{'High volatility — sensitive to shocks.' if volatility > 5 else 'Relatively stable growth pattern.'}"
                )
                st.markdown(
                    f'<div style="background:#1a2a1a; border-left:4px solid #4f8ef7; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                    f'💡 <strong>Takeaway:</strong> {takeaway_text}</div>',
                    unsafe_allow_html=True
                )

                with st.expander("📚 Where it comes from & why it matters"):
                    st.markdown("""
                    **Source:** Federal Housing Finance Agency (FHFA) — the official U.S. government home price index.

                    **Why it matters:** This is our prediction target. Every other variable in FirmScape is evaluated on how well it explains *why this number goes up or down* in a given city.

                    **⚠️ Note:** This measures price *change*, not price *level*.
                    """)
            else:
                st.info("No FHFA data for this city.")

    elif "Signal 1" in selected_var:
        st.markdown("#### 📡 Firm Founding Rate (YoY %)")
        st.info("**What it is:** How much faster or slower new businesses are being founded compared to last year.")

        if use_idf:
            city_pick2 = eda_city_pick
            city_data2 = idf_eda[idf_eda['city_state'] == city_pick2].dropna(subset=['firms_founded_yoy', 'fhfa_yoy']).sort_values(['year', 'quarter'])

            if not city_data2.empty:
                fig, ax = dark_fig()
                ax2 = ax.twinx()
                ax.plot(range(len(city_data2)), city_data2['firms_founded_yoy'].values,
                        color='#f7a44f', linewidth=2, label='Firm Growth %')
                ax2.plot(range(len(city_data2)), city_data2['fhfa_yoy'].values,
                         color='#4f8ef7', linewidth=1.5, linestyle='--', alpha=0.7, label='Housing Price %')
                ax.axhline(0, color='#555', linewidth=0.7)
                yr_rows2 = city_data2.reset_index(drop=True)
                ticks2 = yr_rows2[yr_rows2['quarter'] == 1].index[::4].tolist()
                labels2 = yr_rows2.loc[ticks2, 'year'].astype(str).tolist()
                ax.set_xticks(ticks2); ax.set_xticklabels(labels2, rotation=45, color='white', fontsize=8)
                ax.set_ylabel("Firm Founding Growth %", color='#f7a44f')
                ax2.set_ylabel("Housing Price Change %", color='#4f8ef7')
                ax2.tick_params(colors='white')
                ax2.spines[:].set_color("#333")
                ax2.set_facecolor("#0e1117")
                ax.set_title(f"{city_pick2} — Firm Growth vs Housing Prices", color='white')
                lines1, labels_l1 = ax.get_legend_handles_labels()
                lines2, labels_l2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels_l1 + labels_l2,
                          facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                chart_note("Orange (left axis) = firm founding rate · Blue dashed (right axis) = housing price change")

                insight_df = city_data2[['firms_founded_yoy', 'fhfa_yoy']].dropna()
                if len(insight_df) >= 8:
                    r_now, p_now = pearsonr(insight_df['firms_founded_yoy'], insight_df['fhfa_yoy'])
                    lag_results = {}
                    for lag in [1, 2, 4, 8]:
                        lagged = insight_df['firms_founded_yoy'].shift(lag)
                        combined = pd.DataFrame({'x': lagged, 'y': insight_df['fhfa_yoy']}).dropna()
                        if len(combined) >= 6 and combined['x'].std() > 0 and combined['y'].std() > 0:
                            r_lag, p_lag = pearsonr(combined['x'], combined['y'])
                            lag_results[lag] = (r_lag, p_lag)

                    best_lag = max(lag_results, key=lambda k: abs(lag_results[k][0])) if lag_results else None
                    best_r, best_p = lag_results[best_lag] if best_lag else (r_now, p_now)
                    lag_years = f"{best_lag // 4}yr" if best_lag and best_lag >= 4 else (f"{best_lag}Q" if best_lag else "—")

                    col_i1, col_i2, col_i3 = st.columns(3)
                    col_i1.metric("Same-time correlation (R)", f"{r_now:.2f}")
                    col_i2.metric("Best predictive lag", lag_years)
                    col_i3.metric("Best lag R", f"{best_r:.2f}" if best_lag else "—")

                    sig_label = "significant" if p_now < 0.05 else "not significant"
                    direction = "move together" if r_now > 0.1 else ("move oppositely" if r_now < -0.1 else "weakly correlated")
                    takeaway_firm = (
                        f"Firm growth and housing {direction} (R={r_now:.2f}, {sig_label}). "
                        f"{'📈 Strongest signal at a lag — firm growth leads housing here.' if best_lag and abs(best_r) > abs(r_now) else '⚠️ Weak lead signal for this city.'}"
                    )
                    st.markdown(
                        f'<div style="background:#1a2a1a; border-left:4px solid #f7a44f; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                        f'💡 <strong>Takeaway:</strong> {takeaway_firm}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Not enough data to run lag analysis for this city.")

                with st.expander("📚 Where it comes from & why it matters"):
                    st.markdown("""
                    **Source:** Business registration records, aggregated by city and year.

                    **Why it matters:** New firms = new jobs = new workers moving in = more housing demand. This variable is expected to *lead* housing prices by 1–2 years.
                    """)

    elif "Signal 2" in selected_var:
        st.markdown("#### 📡 Industry Concentration (HHI)")
        st.info("**What it is:** How dominated a city's economy is by a single industry — higher HHI means more economic fragility.")

        if use_idf:
            comp_cities = eda_compare_cities or []
            if comp_cities:
                fig, ax = dark_fig()
                colors_hhi = ['#4f8ef7', '#f7a44f', '#4ff7a4', '#f74f4f']
                for i, city in enumerate(comp_cities[:4]):
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['hhi_new']).sort_values(['year', 'quarter'])
                    if not cd.empty:
                        ax.plot(range(len(cd)), cd['hhi_new'].values, color=colors_hhi[i], linewidth=2, label=city)
                cd_ref = idf_eda[idf_eda['city_state'] == comp_cities[0]].dropna(subset=['hhi_new']).sort_values(['year', 'quarter'])
                yr_rows3 = cd_ref.reset_index(drop=True)
                ticks3 = yr_rows3[yr_rows3['quarter'] == 1].index[::4].tolist() if 'quarter' in yr_rows3.columns else []
                labels3 = yr_rows3.loc[ticks3, 'year'].astype(str).tolist() if ticks3 else []
                if ticks3:
                    ax.set_xticks(ticks3); ax.set_xticklabels(labels3, rotation=45, color='white', fontsize=8)
                ax.set_ylabel("HHI Score", color='white')
                ax.set_title("Industry Concentration Over Time — by City", color='white')
                ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                chart_note("Higher = fewer industries dominate · Lower = more evenly spread economy")

                stat_rows = []
                for city in comp_cities[:4]:
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['hhi_new'])
                    if not cd.empty:
                        stat_rows.append({
                            "City": city,
                            "Avg HHI": f"{cd['hhi_new'].mean():.3f}",
                            "Latest HHI": f"{cd.sort_values('year').iloc[-1]['hhi_new']:.3f}",
                            "Trend": "↑ Rising" if cd.sort_values('year').iloc[-1]['hhi_new'] > cd['hhi_new'].mean() else "↓ Falling"
                        })
                if stat_rows:
                    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
                    most_conc = max(stat_rows, key=lambda x: float(x["Avg HHI"]))
                    st.markdown(
                        f'<div style="background:#1a2a1a; border-left:4px solid #4f8ef7; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                        f'💡 <strong>Takeaway:</strong> <b>{most_conc["City"]}</b> has the highest average concentration.</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("📚 Where it comes from & why it matters"):
                        st.markdown("""
                        **Source:** Calculated from business registration data using the Herfindahl-Hirschman Index (HHI) formula.

                        **Why it matters:** Concentrated cities are fragile — when Detroit's auto industry collapsed, the whole city did too.
                        """)

    elif "Signal 3" in selected_var:
        st.markdown("#### 📡 Industry Diversity (# of Industries)")
        st.info("**What it is:** Count of distinct industry types in a city — more industries means a more resilient local economy.")

        if use_idf and 'industry_count_new' in idf_eda.columns:
            comp_cities2 = eda_compare_cities or []
            if comp_cities2:
                fig, ax = dark_fig()
                colors_div = ['#4f8ef7', '#f7a44f', '#4ff7a4', '#f74f4f']
                for i, city in enumerate(comp_cities2[:4]):
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['industry_count_new']).sort_values(['year', 'quarter'])
                    if not cd.empty:
                        cd_annual = cd.groupby('year')['industry_count_new'].mean().reset_index()
                        ax.plot(cd_annual['year'], cd_annual['industry_count_new'],
                                color=colors_div[i], linewidth=2, marker='o', markersize=3, label=city)
                ax.set_xlabel("Year", color='white')
                ax.set_ylabel("# of distinct industries", color='white')
                ax.set_title("Industry Diversity Over Time — by City", color='white')
                ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                chart_note("Each point = avg # of distinct industries that year · Rising = economy becoming more diverse")

                stat_rows2 = []
                for city in comp_cities2[:4]:
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['industry_count_new'])
                    if not cd.empty:
                        cd_annual = cd.groupby('year')['industry_count_new'].mean()
                        stat_rows2.append({
                            "City": city,
                            "Avg # Industries": f"{cd_annual.mean():.0f}",
                            "Recent # Industries": f"{cd_annual.iloc[-1]:.0f}",
                            "Trend": "↑ Growing" if cd_annual.iloc[-1] > cd_annual.mean() else "↓ Shrinking"
                        })
                if stat_rows2:
                    st.dataframe(pd.DataFrame(stat_rows2), use_container_width=True, hide_index=True)
                    most_div = max(stat_rows2, key=lambda x: float(x["Avg # Industries"]))
                    st.markdown(
                        f'<div style="background:#1a2a1a; border-left:4px solid #4ff7a4; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                        f'💡 <strong>Takeaway:</strong> <b>{most_div["City"]}</b> has the most diverse industry mix.</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("📚 Where it comes from & why it matters"):
                        st.markdown("""
                        **Source:** Count of distinct industry categories per city per year, from business registration data.

                        **Why it matters:** More diverse = more stable. If one industry tanks, others absorb the shock.
                        """)

    elif "Signal 4" in selected_var:
        st.markdown("#### 📡 Top Industry Share (%)")
        st.info("**What it is:** The share of all businesses belonging to the single largest industry in a city.")

        if use_idf and 'top_industry_share_new' in idf_eda.columns:
            comp_cities3 = eda_compare_cities or []
            if comp_cities3:
                fig, ax = dark_fig()
                colors_top = ['#4f8ef7', '#f7a44f', '#4ff7a4', '#f74f4f']
                for i, city in enumerate(comp_cities3[:4]):
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['top_industry_share_new']).sort_values(['year', 'quarter'])
                    if not cd.empty:
                        cd_annual = cd.groupby('year')['top_industry_share_new'].mean().reset_index()
                        ax.plot(cd_annual['year'], cd_annual['top_industry_share_new'] * 100,
                                color=colors_top[i], linewidth=2, marker='o', markersize=3, label=city)
                ax.set_xlabel("Year", color='white')
                ax.set_ylabel("Top industry share (%)", color='white')
                ax.set_title("Top Industry Dominance Over Time — by City", color='white')
                ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
                chart_note("Higher % = one industry dominates more")

                stat_rows3 = []
                for city in comp_cities3[:4]:
                    cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['top_industry_share_new'])
                    if not cd.empty:
                        cd_annual = cd.groupby('year')['top_industry_share_new'].mean() * 100
                        stat_rows3.append({
                            "City": city,
                            "Avg Top Share": f"{cd_annual.mean():.1f}%",
                            "Recent Top Share": f"{cd_annual.iloc[-1]:.1f}%",
                            "Trend": "↑ More dominant" if cd_annual.iloc[-1] > cd_annual.mean() else "↓ Less dominant"
                        })
                if stat_rows3:
                    st.dataframe(pd.DataFrame(stat_rows3), use_container_width=True, hide_index=True)
                    highest = max(stat_rows3, key=lambda x: float(x["Avg Top Share"].rstrip('%')))
                    st.markdown(
                        f'<div style="background:#1a2a1a; border-left:4px solid #f7a44f; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                        f'💡 <strong>Takeaway:</strong> <b>{highest["City"]}</b> has the most dominant single industry.</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("📚 Where it comes from & why it matters"):
                        st.markdown("""
                        **Source:** Calculated from business registration data — largest industry ÷ total businesses.
                        """)

    st.divider()
    st.subheader("🤔 How Do These Variables Predict Housing Prices?")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**1️⃣ Select variables**")
        st.caption("Firm founding rate · Industry concentration · Industry diversity · Top industry share")
    with mc2:
        st.markdown("**2️⃣ Model learns weights**")
        st.caption("Which variables — at which lag — best explain why some cities' housing grew faster?")
    with mc3:
        st.markdown("**3️⃣ Interpret with context**")
        st.caption("Housing depends on interest rates, zoning, supply too. A low R² is expected, not a failure.")
    st.info("💡 Go to **✅ Validation & Modeling** to run the actual models and see which variables come out on top.")

# ─────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────
if tab == "🔎 Evidence":
    st.title("What Patterns Show Up Across Cities?")
    st.markdown("Explore the relationship between industrial clustering and urban housing value.")

    if integrated_df is None:
        import glob
        found_files = glob.glob("*.csv")
        st.error("⚠️ Could not find the integrated dataset.")
        st.code("\n".join(sorted(found_files)) if found_files else "No CSV files found.")
        st.stop()
    else:
        st.caption(f"✅ Loaded: `{integrated_fname}`")

    idf = integrated_df.copy()
    city_col_name = idf['_city_col'].iloc[0] if '_city_col' in idf.columns else 'city_state'
    idf['city_state'] = idf[city_col_name].astype(str)

    # Build city list: curated 5 pinned at top, then rest of top-100
    default_cities = CURATED_CITIES if CURATED_CITIES else TOP_100_CITIES[:5]
    remaining_cities = [c for c in TOP_100_CITIES if c not in default_cities]
    city_options = default_cities + remaining_cities  # curated 5 always appear first

    # ── 1. CITY TIMELINE ─────────────────────────────────────────────────────
    st.subheader("🏙️ City Timeline: Housing Price & Firm Growth")
    st.caption("Uses quarterly FHFA / Zillow housing data and firm founding rates from 1977–present.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        preset_city = st.session_state.get('preset_city')
        # Default to first curated city unless a preset has been loaded
        default_city_idx = 0
        if preset_city:
            if preset_city in city_options:
                default_city_idx = city_options.index(preset_city)
            else:
                for i, c in enumerate(city_options):
                    if preset_city.split(",")[0].lower() in c.lower():
                        default_city_idx = i
                        break
        city_choice = st.selectbox("Select a City:", city_options, index=default_city_idx, key="ev_city")

    with col_c2:
        housing_label_map = {
            "📈 % change in home prices vs. last year (FHFA)": "fhfa_yoy",
            "🏠 Raw home price score over time (FHFA)":        "fhfa_index",
            "📈 % change in home prices vs. last year (Zillow)": "zillow_yoy",
            "💵 Actual median home price in dollars (Zillow)": "zillow_price_q",
        }
        preset_metric = st.session_state.get('preset_housing_metric')
        default_metric_idx = 0
        if preset_metric:
            for i, label in enumerate(housing_label_map.keys()):
                if housing_label_map[label] in str(preset_metric):
                    default_metric_idx = i
                    break
        housing_metric_label = st.selectbox(
            "Housing Metric:", list(housing_label_map.keys()),
            index=default_metric_idx, key="ev_housing"
        )
        h_col = housing_label_map[housing_metric_label]

    # Clear preset banner if user manually picked a different city
    if st.session_state.get('preset_city') and city_choice != st.session_state.get('preset_city'):
        st.session_state['case_study_preset'] = None
        st.session_state['preset_city'] = None
        st.session_state['preset_housing_metric'] = None

    if st.session_state.get('case_study_preset'):
        active_cs = st.session_state['case_study_preset']
        active_preset = CASE_STUDY_PRESETS[active_cs]
        st.info(
            f"📍 **{active_cs} preset active** — {active_preset['emoji']} "
            f"*{active_preset['what_to_look_for']}*"
        )

    city_ts = idf[idf['city_state'] == city_choice].sort_values(['year', 'quarter'])

    if city_ts.empty:
        st.warning(f"No data for {city_choice}.")
    else:
        city_ts = city_ts.dropna(subset=[h_col], how='all')
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=city_ts['yq'], y=city_ts[h_col],
            name=housing_metric_label,
            line=dict(color='#4f8ef7', width=2),
            fill='tozeroy',
            fillcolor='rgba(79,142,247,0.1)'
        ))
        fig_ts.update_layout(
            title=f"{city_choice} — Home Price Change Over Time",
            template='plotly_dark',
            yaxis=dict(title=housing_metric_label, titlefont=dict(color='#4f8ef7')),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=400,
            xaxis=dict(
                title=dict(text="Quarter (Year + Quarter)", font=dict(color='white')),
                tickmode='array',
                tickvals=city_ts['yq'].iloc[::16].tolist(),
                tickangle=45
            )
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        hvals = city_ts[h_col].dropna()
        if len(hvals) > 0:
            is_pct_metric = h_col in ('fhfa_yoy', 'zillow_yoy')
            if is_pct_metric:
                avg_chg = hvals.mean() * 100
                recent = hvals.iloc[-4:].mean() * 100
                pct_pos = (hvals > 0).mean() * 100
                trend = "accelerating" if recent > avg_chg else "slowing"
                takeaway_ts = (
                    f"Prices were rising in <b>{pct_pos:.0f}%</b> of quarters, "
                    f"averaging <b>{avg_chg:.1f}% per year</b> — "
                    f"the most recent year is <b>{trend}</b> vs. the long-run average."
                )
            else:
                first_val = hvals.iloc[0]
                last_val = hvals.iloc[-1]
                total_chg = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0
                direction = "risen" if total_chg > 0 else "fallen"
                takeaway_ts = (
                    f"Since the first recorded period, home prices have <b>{direction} {abs(total_chg):.0f}%</b> overall."
                )
            st.markdown(
                f'<div style="background:#1a2a1a; border-left:4px solid #4f8ef7; border-radius:6px; padding:10px 14px; margin:4px 0 12px 0;">'
                f'💡 <strong>Takeaway:</strong> {takeaway_ts}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ── 2. RELATIONSHIP CHART ─────────────────────────────────────────────────
    st.subheader("📊 How Do Industry Variables Relate to Housing Prices?")
    st.caption("Each dot = one city (averaged across all years). The orange line shows the overall trend. **R² here is cross-city — it does not change with the city selected above.**")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        x_label_map = {
            "🏭 How dominated is the city by one industry? (HHI)": "hhi_new",
            "📊 What % of firms are in the #1 industry?":          "top_industry_share_new",
            "🌐 How many different industries exist in the city?":  "industry_count_new",
            "📈 How fast are new businesses being founded? (YoY)": "firms_founded_yoy",
        }
        x_axis_label = st.selectbox("Industry variable:", list(x_label_map.keys()), key="sc_x")
        x_col = x_label_map[x_axis_label]

    with sc_col2:
        y_label_map = {
            "💵 Avg home price in dollars (Zillow)":           "zillow_price_q",
            "📈 Avg annual home price change % (FHFA)":        "fhfa_yoy",
        }
        y_axis_label = st.selectbox("Housing variable:", list(y_label_map.keys()), key="sc_y")
        y_col = y_label_map[y_axis_label]

    sc_df = idf.groupby('city_state')[[x_col, y_col]].mean().dropna().reset_index()
    x_p1, x_p99 = sc_df[x_col].quantile([0.01, 0.99])
    y_p1, y_p99 = sc_df[y_col].quantile([0.01, 0.99])
    sc_df = sc_df[(sc_df[x_col].between(x_p1, x_p99)) & (sc_df[y_col].between(y_p1, y_p99))]

    if not sc_df.empty:
        is_pct = y_col == 'fhfa_yoy'
        y_vals = sc_df[y_col] * 100 if is_pct else sc_df[y_col]

        fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
        fig_sc.patch.set_facecolor("#0e1117")
        ax_sc.set_facecolor("#0e1117")
        ax_sc.scatter(sc_df[x_col], y_vals, color='#4f8ef7', alpha=0.6, s=40, linewidths=0)

        if len(sc_df) >= 5:
            m, b = np.polyfit(sc_df[x_col], y_vals, 1)
            x_line = np.linspace(sc_df[x_col].min(), sc_df[x_col].max(), 200)
            ax_sc.plot(x_line, m * x_line + b, color='#f7a44f', linewidth=2, linestyle='--', label='Trend')
            ax_sc.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=9)

        x_label_short = x_axis_label.split("(")[0].strip().lstrip("🏭📊🌐📈").strip()
        x_unit = "(0 = perfectly diverse, 1 = one industry dominates)" if x_col == "hhi_new" else \
                 "(0% = no dominance, 100% = one industry has all firms)" if x_col == "top_industry_share_new" else \
                 "(count of distinct industry types)" if x_col == "industry_count_new" else \
                 "(0 = no growth, 0.5 = 50% more firms than last year)"
        y_label_short = "Avg home price ($)" if not is_pct else "Avg annual price change (%/yr)"
        ax_sc.set_xlabel(f"{x_label_short}\n{x_unit}", color='white', fontsize=9)
        ax_sc.set_ylabel(y_label_short, color='white', fontsize=10)
        ax_sc.set_title(f"{x_label_short} vs. {y_label_short}", color='white', fontsize=12)
        ax_sc.tick_params(colors='white')
        ax_sc.spines[:].set_color('#333')
        ax_sc.axhline(0, color='#555', linewidth=0.8, linestyle=':')
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        sc_clean = sc_df[[x_col, y_col]].dropna()
        if len(sc_clean) >= 5 and sc_clean[x_col].std() > 0 and sc_clean[y_col].std() > 0:
            r_sc, p_sc = pearsonr(sc_clean[x_col], sc_clean[y_col])
            sig = "significant" if p_sc < 0.05 else "not significant"
            x_descriptions = {
                "hhi_new":               "more economically concentrated cities (dominated by one industry)",
                "top_industry_share_new": "cities where one industry makes up a larger share of all firms",
                "industry_count_new":    "cities with more diverse industry types",
                "firms_founded_yoy":     "cities where new businesses are being founded faster",
            }
            x_desc = x_descriptions.get(x_col, f"cities with more {x_label_short.lower()}")
            y_desc = "home prices" if not is_pct else "annual home price growth"
            higher_lower = "higher" if r_sc > 0 else "lower"
            sig_phrase = "a statistically significant finding" if p_sc < 0.05 else "not statistically significant — could be noise"
            sc_c1, sc_c2, sc_c3 = st.columns(3)
            sc_c1.metric("Correlation (R)", f"{r_sc:.3f}")
            sc_c2.metric("Strength (R²)", f"{r_sc**2:.3f}")
            sc_c3.metric("Statistically significant?", "Yes ✅" if p_sc < 0.05 else "No ❌")
            takeaway_sc = (
                f"On average, <b>{x_desc}</b> tend to have <b>{higher_lower} {y_desc}</b> "
                f"(R={r_sc:.2f}) — {sig_phrase}."
            )
            st.markdown(
                f'<div style="background:#1a2a1a; border-left:4px solid #4f8ef7; border-radius:6px; padding:10px 14px; margin:8px 0;">'
                f'💡 <strong>Takeaway:</strong> {takeaway_sc}</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("No data available for this combination.")


    st.divider()



    # ── 3. MULTI-CITY HOUSING TREND ───────────────────────────────────────────
    st.subheader("📈 Compare Housing Trends Across Cities")

    mc_cities = st.multiselect(
        "Select cities to compare (up to 5):",
        city_options,          # curated 5 appear first in the dropdown
        default=default_cities,  # all 5 curated cities pre-selected
        key="mc_cities"
    )
    mc_metric_map = {
        "📈 % change in home prices vs. last year (FHFA)":   "fhfa_yoy",
        "🏠 Raw home price score over time (FHFA)":          "fhfa_index",
        "📈 % change in home prices vs. last year (Zillow)": "zillow_yoy",
        "💵 Actual median home price in dollars (Zillow)":   "zillow_price_q",
    }
    mc_metric_label = st.selectbox("Housing metric to compare:", list(mc_metric_map.keys()), key="mc_metric")
    mc_metric = mc_metric_map[mc_metric_label]

    if mc_cities:
        mc_df = idf[idf['city_state'].isin(mc_cities[:5])].dropna(subset=[mc_metric])
        mc_df = mc_df.sort_values(['city_state', 'year', 'quarter'])
        fig_mc = px.line(
            mc_df, x='yq', y=mc_metric,
            color='city_state',
            labels={'yq': 'Quarter', mc_metric: mc_metric, 'city_state': 'City'},
            template='plotly_dark',
            title=f"{mc_metric_label} — Multi-City Comparison",
            height=400
        )
        all_yqs = sorted(mc_df['yq'].unique())
        tick_yqs = all_yqs[::16]
        fig_mc.update_xaxes(tickmode='array', tickvals=tick_yqs, tickangle=45)
        st.plotly_chart(fig_mc, use_container_width=True)

    st.divider()

    # ── 4. INFLECTION POINT DETECTOR ─────────────────────────────────────────
    st.subheader("🎯 Automated Insights")
    if st.button("✨ Spot the Inflection Point"):
        city_ts_inf = idf[idf['city_state'] == city_choice].sort_values(['year', 'quarter'])
        city_ts_inf = city_ts_inf.dropna(subset=['firms_founded_yoy', 'fhfa_yoy'])

        if len(city_ts_inf) < 8:
            st.info(f"Not enough quarterly data for {city_choice} to detect an inflection point.")
        else:
            city_ts_inf = city_ts_inf.copy()
            city_ts_inf['firm_accel'] = city_ts_inf['firms_founded_yoy'].diff()
            idx_max = city_ts_inf['firm_accel'].idxmax()
            inf_row = city_ts_inf.loc[idx_max]
            inf_yq = inf_row['yq']
            inf_year = int(inf_row['year'])

            growth_before = city_ts_inf[city_ts_inf['year'] < inf_year]['firms_founded_yoy'].mean()
            growth_after = city_ts_inf[city_ts_inf['year'] >= inf_year]['firms_founded_yoy'].mean()

            post = city_ts_inf[city_ts_inf['year'] >= inf_year].dropna(subset=['firms_founded_yoy', 'fhfa_yoy'])
            if len(post) >= 4 and post['firms_founded_yoy'].std() > 0 and post['fhfa_yoy'].std() > 0:
                r_post, p_post = pearsonr(post['firms_founded_yoy'], post['fhfa_yoy'])
                r2_post = round(r_post ** 2, 3)
                p_label = f"{p_post:.4f}"
            else:
                r2_post, p_label = "n/a", "n/a"

            st.balloons()
            st.success(f"**Inflection Detected:** Firm growth in **{city_choice}** accelerated most sharply around **{inf_yq}**.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Firm Growth Before", f"{growth_before:.1f}%")
            c2.metric("Firm Growth After", f"{growth_after:.1f}%")
            c3.metric("Post-Inflection R²", str(r2_post))
            st.caption(f"P-value after inflection: {p_label}.")

            fig_inf, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(range(len(city_ts_inf)), city_ts_inf['firms_founded_yoy'].values,
                    color='#f7a44f', linewidth=2, label='Firm Growth YoY %')
            ax.plot(range(len(city_ts_inf)), city_ts_inf['fhfa_yoy'].values,
                    color='#4f8ef7', linewidth=2, label='FHFA YoY %', alpha=0.8)
            inf_pos = city_ts_inf.index.get_loc(idx_max)
            ax.axvline(inf_pos, color='yellow', linestyle='--', linewidth=1.5, label=f'Inflection ({inf_yq})')
            ax.axvspan(inf_pos, min(inf_pos + 8, len(city_ts_inf) - 1),
                       color='yellow', alpha=0.12, label='Post-Inflection Window')
            year_ticks = city_ts_inf.reset_index(drop=True)
            tick_positions = year_ticks[year_ticks['quarter'] == 1].index[::4].tolist()
            tick_labels = year_ticks.loc[tick_positions, 'year'].astype(str).tolist()
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, color='white', fontsize=8)
            ax.set_facecolor("#0e1117")
            fig_inf.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            ax.set_ylabel("% Change", color="white")
            ax.set_title(f"{city_choice} — Firm Growth & Housing Price YoY", color="white")
            ax.legend(facecolor="#0e1117", labelcolor="white", fontsize=8)
            st.pyplot(fig_inf)

# ─────────────────────────────────────────────
# VALIDATION & MODELING TAB
# ─────────────────────────────────────────────
if tab == "✅ Validation & Modeling":
    st.title("✅ Validation & Interactive Modeling")
    st.markdown(
        "Test how well industry variables predict housing price changes. "
        "**A low R² is expected** — housing prices have many drivers. "
        "Our goal is to find *which variables matter most*, not to overfit."
    )

    st.subheader("Step 1: Choose a Model")
    model_choice = st.selectbox(
        "Select a model to run:",
        ["Linear Regression (Interpretable)", "XGBoost (Best Performance)", "Random Forest (Feature Importance)"],
        key="model_selector"
    )

    st.subheader("Step 2: Choose a Hypothesis")
    hypothesis = st.selectbox(
        "What relationship are we testing?",
        [
            "Does industry growth lead housing growth?",
            "Do concentrated cities have higher housing volatility?",
            "Do diverse cities grow more steadily?"
        ],
        key="hyp_selector"
    )

    st.subheader("Step 3: Lag & Significance Testing")
    col_v1, col_v2 = st.columns([1, 2])

    with col_v1:
        lag_years = st.slider("Lag Housing Data by (Years):", 0, 3, 1, key="lag_slider")
        sig_filter = st.toggle("Show only significant results (p < 0.05)", value=True, key="sig_toggle")

    val_df = merged_companies_housing.copy()
    val_df = val_df.sort_values(['city_x', 'year_int'])
    val_df['lagged_housing'] = val_df.groupby('city_x')['pct_change'].shift(-lag_years)
    clean_val = val_df.dropna(subset=['lagged_housing', 'pct_change'])

    if not clean_val.empty:
        x_arr = clean_val['pct_change'].values
        y_arr = clean_val['lagged_housing'].values
        if x_arr.std() > 0 and y_arr.std() > 0:
            slope, intercept, r_val, p_val, std_err = stats.linregress(x_arr, y_arr)
        else:
            slope, intercept, r_val, p_val, std_err = 0, 0, 0, 1, 0
        r2 = r_val ** 2

        with col_v2:
            st.write(f"**Lag Results for {lag_years} Year(s):**")
            if sig_filter and p_val >= 0.05:
                st.warning("No statistically significant relationship found for this lag.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Correlation (R)", f"{r_val:.3f}")
                m2.metric("P-Value", f"{p_val:.4f}")
                m3.metric("R² (Variance Explained)", f"{r2:.3f}")
                st.write(f"Sample Size: **{len(clean_val):,} data points**")

    st.divider()
    st.subheader("Step 4: Practical Significance")
    if not clean_val.empty and (not sig_filter or p_val < 0.05):
        st.info(f"""
        **The 'So What?' Factor:** A **10% increase** in company growth is associated with a 
        **{10 * slope:.2f}%** change in housing growth **{lag_years} year(s)** later.

        **R² = {r2:.3f}** — Industry concentration explains roughly **{r2*100:.1f}%** of housing 
        price variance. The remaining {(1-r2)*100:.1f}% is driven by other factors.
        """)

    st.divider()
    st.subheader("Step 5: Run the Model Interactively")

    if st.button("🚀 Run Model"):
        

        model_df = clean_val.copy()
        le = LabelEncoder()
        if 'industry' in model_df.columns:
            model_df['industry_enc'] = le.fit_transform(model_df['industry'].fillna("Unknown"))
        else:
            model_df['industry_enc'] = 0

        div_df = (
            merged_companies_housing.groupby(["city_x", "year_int"])["industry"]
            .nunique()
            .reset_index()
            .rename(columns={"industry": "diversity"})
        ) if 'industry' in merged_companies_housing.columns else pd.DataFrame(columns=["city_x","year_int","diversity"])
        model_df = model_df.merge(div_df, on=["city_x", "year_int"], how="left")
        model_df['diversity'] = model_df['diversity'].fillna(1)

        features = ['pct_change', 'industry_enc', 'diversity']
        target = 'lagged_housing'

        X = model_df[features].dropna()
        y = model_df.loc[X.index, target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if "XGBoost" in model_choice:
            try:
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
                model_label = "XGBoost"
            except Exception:
                model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
                model_label = "Gradient Boosting (XGBoost fallback)"
        elif "Random Forest" in model_choice:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model_label = "Random Forest"
        else:
            model = LinearRegression()
            model_label = "Linear Regression"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.success(f"**{model_label} Results:**")
        rc1, rc2 = st.columns(2)
        rc1.metric("R² Score", f"{r2:.4f}")
        rc2.metric("RMSE", f"{rmse:.4f}")
        st.caption(f"💡 R² = {r2:.3f} means industry variables explain ~{r2*100:.1f}% of housing price variance.")

        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({
                "Feature": ["Company Growth %", "Industry Type", "Industry Diversity"],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                template="plotly_dark",
                title="Feature Importance — Which Variables Drive Housing Price Changes?")
            st.plotly_chart(fig_fi, use_container_width=True)
        elif model_label == "Linear Regression":
            coef_df = pd.DataFrame({
                "Feature": ["Company Growth %", "Industry Type", "Industry Diversity"],
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=True)
            fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                color="Coefficient", color_continuous_scale="RdBu",
                template="plotly_dark", title="Linear Regression Coefficients")
            st.plotly_chart(fig_coef, use_container_width=True)

    st.divider()
    st.subheader("Step 6: Compare Curated Cities")
    st.caption("Comparing the top data-rich cities matching each iconic case study.")

    selected_cities = st.multiselect(
        "Select cities to compare (max 5):",
        CURATED_CITIES,
        default=CURATED_CITIES[:3]
    )

    if selected_cities:
        city_results = []
        for city in selected_cities[:5]:
            city_key = city.split(",")[0]
            data_c = clean_val[clean_val['city_x'].str.contains(city_key, case=False, na=False)]
            if len(data_c) >= 3:
                try:
                    x_vals = data_c['pct_change'].values
                    y_vals = data_c['lagged_housing'].values
                    if x_vals.std() == 0 or y_vals.std() == 0:
                        raise ValueError("Constant array")
                    r_c, p_c = pearsonr(x_vals, y_vals)
                    city_results.append({
                        "City": city,
                        "Lag Correlation (R)": round(r_c, 3),
                        "R²": round(r_c ** 2, 3),
                        "P-Value": round(p_c, 4),
                        "Data Points": len(data_c),
                        "Significant": "✅" if p_c < 0.05 else "❌"
                    })
                except Exception:
                    city_results.append({
                        "City": city, "Lag Correlation (R)": "n/a",
                        "R²": "n/a", "P-Value": "n/a",
                        "Data Points": len(data_c), "Significant": "⚠️ No variance"
                    })
            else:
                city_results.append({
                    "City": city, "Lag Correlation (R)": "n/a",
                    "R²": "n/a", "P-Value": "n/a", "Data Points": len(data_c), "Significant": "⚠️ Too few"
                })
        if city_results:
            st.dataframe(pd.DataFrame(city_results), use_container_width=True)

# ─────────────────────────────────────────────
# OPPORTUNITY LAB TAB
# ─────────────────────────────────────────────
if tab == "🚀 Opportunity Lab":
    st.title("🚀 Opportunity Lab: The 'Next Hub' Finder")
    st.markdown(
        "Build your own investment shortlist by weighting the economic signals that matter most to you."
    )

    if integrated_df is not None:
        opp_df_source = integrated_df.copy()
        city_col = 'city_state'
        growth_col = 'firms_founded_yoy'
        div_col = 'industry_count_new'
        st.success(f"✅ Using `{integrated_fname}`")
    else:
        opp_df_source = merged_companies_housing.copy()
        city_col = 'city_x'
        growth_col = 'pct_change'
        div_col = None
        st.warning("⚠️ Integrated dataset not found — using merged_companies_housing fallback.")

    with st.sidebar:
        st.header("⚖️ Strategy Weights")
        w_growth = st.slider("Company Growth", 0.0, 1.0, 0.4, key="w_growth")
        w_diversity = st.slider("Industry Diversity", 0.0, 1.0, 0.3, key="w_div")
        w_afford = st.slider("Housing Affordability", 0.0, 1.0, 0.2, key="w_aff")
        w_stability = st.slider("Price Stability", 0.0, 1.0, 0.1, key="w_stab")
        st.divider()
        lookback = st.slider("Analysis Window (Years)", 1, 5, 3, key="lookback")

    opp_df = opp_df_source.copy()
    if integrated_df is not None:
        city_col_name2 = opp_df['_city_col'].iloc[0] if '_city_col' in opp_df.columns else 'city_state'
        opp_df['city_state'] = opp_df[city_col_name2].astype(str)
        city_col = 'city_state'

    if div_col and div_col in opp_df.columns:
        city_stats = opp_df.groupby(city_col).agg(
            growth=(growth_col, 'mean'),
            diversity=(div_col, 'mean')
        ).reset_index()
    else:
        city_stats = opp_df.groupby(city_col).agg(
            growth=(growth_col, 'mean')
        ).reset_index()
        city_stats['diversity'] = 1

    for col in ['growth', 'diversity']:
        rng = city_stats[col].max() - city_stats[col].min()
        if rng > 0:
            city_stats[col + '_norm'] = (city_stats[col] - city_stats[col].min()) / rng
        else:
            city_stats[col + '_norm'] = 0.5

    city_stats['Final_Score'] = (
        city_stats['growth_norm'] * w_growth +
        city_stats['diversity_norm'] * w_diversity
    ) * 100

    st.subheader("🏆 The 'Next Hub' Leaderboard")
    top_10 = city_stats.sort_values('Final_Score', ascending=False).head(10)
    fig_lead = px.bar(
        top_10, x='Final_Score', y=city_col, orientation='h',
        color='Final_Score', color_continuous_scale='Viridis',
        labels={city_col: 'City', 'Final_Score': 'Opportunity Score'},
        template="plotly_dark"
    )
    fig_lead.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
    st.plotly_chart(fig_lead, use_container_width=True)
    st.caption("📌 Scores reflect a weighted combination of company growth and industry diversity.")

    st.divider()
    st.subheader("⚡ What-If Shock Simulator")
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        target_city = st.selectbox("Pick a City to Shock:", top_10[city_col].tolist())
        shock_type = st.selectbox("Scenario:", [
            "+20% Tech Growth",
            "Manufacturing Decline (-15%)",
            "Housing Spike (+10%)"
        ])
        if st.button("Run Simulation"):
            base_score = city_stats.loc[city_stats[city_col] == target_city, 'Final_Score'].values[0]
            sim_score = base_score * 1.2 if "+" in shock_type else base_score * 0.85
            with col_s2:
                st.write(f"### Result for {target_city}")
                st.metric("Simulated Score", f"{sim_score:.1f}", delta=f"{sim_score - base_score:.1f}")
                st.write(f"This shock would move **{target_city}** on the leaderboard relative to current weights.")

    st.divider()
    if st.button("🎯 Generate Judge's Shortlist"):
        st.subheader("Top 3 High-Conviction Cities")
        final_3 = top_10.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(final_3.iterrows()):
            city_name = row[city_col]
            with cols[i]:
                st.success(f"**{i+1}. {city_name}**")
                st.write(f"**Score:** {row['Final_Score']:.1f}")
                st.write(f"📈 **Growth:** {row['growth']:.2f}% avg")
                st.write(f"🏭 **Diversity:** {row['diversity']:.0f} industries")
                st.write(f"✅ **Why:** High industrial momentum.")
                st.write(f"⚠️ **Risk:** Use as screening tool — validate with macro factors.")
        st.caption("💡 These cities score highest given your current weight settings.")
