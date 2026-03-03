import os
# ─── REQUIRED PACKAGES ──────────────────────────────────────────────────────
# If you see Pylance "cannot be resolved" warnings, your VS Code interpreter
# is not pointing to the right environment. Fix:
#   1. Open Terminal in VS Code
#   2. Run: pip install streamlit pandas matplotlib seaborn numpy scipy plotly scikit-learn
#   3. Press Cmd+Shift+P → "Python: Select Interpreter" → pick the one with these packages
#   4. pip install xgboost  (optional — falls back to GradientBoosting if missing)
# The app WILL run fine even with Pylance warnings — they are type-checker false alarms.
# ────────────────────────────────────────────────────────────────────────────
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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("merged_companies_housing.csv")

    # ── Robust year extraction: handle many date formats ──────────────────────
    # Try standard datetime parse first
    year_series = pd.to_datetime(df['date'], errors='coerce').dt.year

    # If that failed for most rows, try extracting 4-digit year from string
    # e.g. "2015-Q2", "2015Q2", "2015", "01/2015", etc.
    if year_series.isna().mean() > 0.5:
        year_series = df['date'].astype(str).str.extract(r'(\b\d{4}\b)')[0].astype(float)

    # If there's a separate 'year' column, use that instead
    if 'year' in df.columns and df['year'].notna().mean() > 0.5:
        year_series = pd.to_numeric(df['year'], errors='coerce')

    df['year_int'] = year_series.astype('Int64')   # nullable int keeps NaT as NA
    df = df.dropna(subset=['year_int'])
    df['year_int'] = df['year_int'].astype(int)
    # ──────────────────────────────────────────────────────────────────────────

    df['city_x'] = df['city_x'].fillna("Unknown").astype(str)
    df = df.dropna(subset=['pct_change'])
    return df

@st.cache_data
def load_integrated_data():
    """Load the quarterly integrated dataset — tries several possible filenames."""
    possible_names = [
        "firmscape_integrated_cbsa_quarterly_cleaned.csv",
        "firmscape_integrated_cbsa.csv",
        "firmscape_integrated_cbsa_quarterly.csv",
        "firmscape_active_businesses.csv",
        "firmscape_active_businesse.csv",  # truncated name visible in VS Code
    ]
    # Also try any CSV in the folder that starts with "firmscape_integrated"
    import glob
    found = glob.glob("firmscape_integrated*.csv") + glob.glob("firmscape_active*.csv")
    all_names = possible_names + [f for f in found if f not in possible_names]

    for fname in all_names:
        try:
            df = pd.read_csv(fname)
            # Detect city column
            for city_candidate in ['city_state', 'city', 'metro_name', 'cbsa_name']:
                if city_candidate in df.columns:
                    df[city_candidate] = df[city_candidate].fillna("Unknown").astype(str)
                    df['_city_col'] = city_candidate  # store which col to use
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
            return df, fname  # return df AND the filename that worked
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
# CURATED CITIES (professor said 3-5 familiar cities)
# ─────────────────────────────────────────────
CURATED_CITIES = ["Detroit, MI", "Austin, TX", "San Jose, CA", "Seattle, WA", "Chicago, IL"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("FirmScape Dashboard")

# Stakeholder selector (professor's feedback: different audiences)
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
# SESSION STATE — for case study presets
# ─────────────────────────────────────────────
CASE_STUDY_PRESETS = {
    "Detroit Manufacturing": {
        "city": "Detroit, MI",
        "housing_metric": "fhfa_yoy (FHFA % change YoY)",
        "story": "Detroit's manufacturing dominance peaked in the 1970s. As auto industry employment collapsed through the 80s–90s, housing prices followed — with a lag. Watch how firm founding rates dried up *before* the housing crash.",
        "what_to_look_for": "A sharp decline in `firms_founded_yoy` in the early 1980s, followed by falling `fhfa_yoy` 2–4 years later.",
        "emoji": "🏭",
    },
    "Bay Area Technology": {
        "city": "San Jose, CA",
        "housing_metric": "fhfa_yoy (FHFA % change YoY)",
        "story": "Silicon Valley's tech boom drove some of the most extreme housing appreciation in U.S. history. Firm concentration (HHI) is high — a handful of industries dominate — yet housing prices kept climbing.",
        "what_to_look_for": "High `top_industry_share_new` + rising `fhfa_yoy` in the 1990s dot-com era and again post-2012.",
        "emoji": "💻",
    },
    "Austin Technology": {
        "city": "Austin, TX",
        "housing_metric": "fhfa_yoy (FHFA % change YoY)",
        "story": "Austin transformed from a government/university city to a tech hub through the 2010s. Firm diversity grew, new companies flooded in, and housing prices exploded — especially post-2020.",
        "what_to_look_for": "Rising `industry_count_new` and `firms_founded_yoy` from 2010 onward, with housing lagging by ~2 years.",
        "emoji": "🤠",
    },
    "Seattle Tech": {
        "city": "Seattle, WA",
        "housing_metric": "fhfa_yoy (FHFA % change YoY)",
        "story": "Amazon and Microsoft anchored Seattle's tech transformation. High industry concentration (few dominant firms) paired with explosive housing growth — a case study in how a single industry can reshape a city.",
        "what_to_look_for": "High `hhi_new` (concentration) paired with rising housing prices — the opposite of the 'diversity = stability' hypothesis.",
        "emoji": "☁️",
    },
}

# Initialize session state for case study preset
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

    # ── WHO IT'S FOR + WHAT VALUE IT BRINGS ──────────────────────────────────
    st.title("FirmScape")
    st.markdown("##### *Industry shifts move housing prices. We built the tool to see it coming.*")
    st.divider()

    # Customer + Value statement — the professor's ask
    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("""
        **We're building FirmScape for two customers:**

        🏠 **Housing investors & urban analysts** who want to know *which cities are about to heat up* — 
        before the housing market prices it in.

        📊 **Business strategists & city planners** who need to understand *how industry clustering 
        shapes economic trajectories* — and where to place their next bet.

        **The value we deliver:**
        > *When industries cluster and grow in a city, how do housing prices move — now and later?*  
        > FirmScape turns 50 years of firm founding and housing data into a single, interactive signal.
        """)
    with c_right:
        st.markdown("####")
        st.metric("Cities / metros", "13,330")
        st.metric("Years of data", "50")
        st.metric("Companies tracked", "275,438")
        st.metric("Data coverage", "96%")

    st.divider()

    # ── HONEST MODEL FRAMING ──────────────────────────────────────────────────
    with st.expander("⚠️ What this model does — and doesn't — claim"):
        st.markdown("""
        Housing prices are driven by **many factors** (interest rates, zoning, supply, demographics).  
        FirmScape focuses on one question: **how much does industry structure explain?**

        - A **low R²** is *expected and honest* — industry data is one signal among many
        - We don't chase high p-values at the cost of validity
        - Use FirmScape as a **leading indicator**, not a complete forecast
        - Goal: identify *which industry variables are most predictive*, then layer in other factors
        """)

    st.divider()

    # ── CASE STUDY EXPLORER — the interactive piece ───────────────────────────
    st.subheader("🗺️ Explore a Famous Case Study")
    st.markdown("Pick a city whose story you know — then follow the data to see if the numbers match the narrative.")

    cs_cols = st.columns(4)
    cs_names = list(CASE_STUDY_PRESETS.keys())

    # Card buttons for each case study
    selected_cs = st.selectbox(
        "Pick a famous case study:",
        cs_names,
        index=0,
        key="home_case_study"
    )

    preset = CASE_STUDY_PRESETS[selected_cs]

    # Show the case study card
    st.markdown(f"""
    <div style="background: #1a1f2e; border-left: 4px solid #4f8ef7; border-radius: 8px; padding: 20px; margin: 12px 0;">
        <h3 style="margin:0; color:white;">{preset['emoji']} {selected_cs}</h3>
        <p style="color: #aab4c8; margin: 10px 0 6px 0;">{preset['story']}</p>
        <p style="color: #f7a44f; font-size: 0.88em; margin:0;">
            📌 <strong>What to look for:</strong> {preset['what_to_look_for']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # The "Go" button stores the preset in session state
    if st.button(f"🚀 Load {selected_cs} preset → jump to Evidence tab", type="primary"):
        st.session_state['case_study_preset'] = selected_cs
        st.session_state['preset_city'] = preset['city']
        st.session_state['preset_housing_metric'] = preset['housing_metric']
        st.success(
            f"✅ Preset loaded! Navigate to **🔎 Evidence** in the sidebar — "
            f"it's now pre-set to **{preset['city']}** with **{preset['housing_metric']}**."
        )

    if st.session_state.get('case_study_preset'):
        active = st.session_state['case_study_preset']
        st.info(f"📍 Active preset: **{active}** → {CASE_STUDY_PRESETS[active]['city']}. Go to **🔎 Evidence** to explore.")

    st.divider()

    # ── HOW TO USE ────────────────────────────────────────────────────────────
    st.subheader("How to use FirmScape")
    hw1, hw2, hw3, hw4 = st.columns(4)
    with hw1:
        st.markdown("**1️⃣ EDA Explorer**")
        st.caption("Understand each variable — what it is, where it comes from, why it matters for housing prices.")
    with hw2:
        st.markdown("**2️⃣ Evidence**")
        st.caption("City timelines, scatter plots, and multi-city comparisons powered by 50 years of quarterly data.")
    with hw3:
        st.markdown("**3️⃣ Validation & Modeling**")
        st.caption("Run XGBoost, Random Forest, or Linear Regression — see which variables are truly predictive.")
    with hw4:
        st.markdown("**4️⃣ Opportunity Lab**")
        st.caption("Weight the signals that matter to you and generate a 'Next Hub' shortlist.")

    st.caption("*Disclaimer: No causal claims — all findings are associative and data-driven.*")

# ─────────────────────────────────────────────
# BUILD THE DATASET TAB
# ─────────────────────────────────────────────
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
        "Each variable we use tells a piece of the story. Pick one below to learn "
        "**what it measures**, **where it comes from**, and **how it relates to housing prices** — "
        "explained simply, shown with real data."
    )

    # ── Variable selector ─────────────────────────────────────────────────────
    selected_var = st.selectbox("🔍 Select a variable to explore:", [
        "🏠 Housing Price Change (FHFA YoY %) — our TARGET",
        "🏗️ Firm Founding Rate (YoY %) — how fast companies are being born",
        "🏭 Industry Concentration (HHI) — how dominated a city is by one industry",
        "🌐 Industry Diversity — how many different industries exist in a city",
        "📊 Top Industry Share — how much the #1 industry controls",
    ], key="eda_var")

    st.divider()

    # ── Use integrated dataset if available, else fallback ────────────────────
    use_idf = integrated_df is not None
    if use_idf:
        idf_eda = integrated_df.copy()
        city_col_eda = idf_eda['_city_col'].iloc[0] if '_city_col' in idf_eda.columns else 'city_state'
        idf_eda['city_state'] = idf_eda[city_col_eda].astype(str)
        all_eda_cities = sorted(idf_eda['city_state'].unique())
        PREFERRED_EDA = ["Austin, TX", "Detroit, MI", "San Jose, CA", "Seattle, WA", "Chicago, IL"]
        eda_cities = [c for c in PREFERRED_EDA if c in all_eda_cities] or all_eda_cities[:5]
    else:
        idf_eda = None
        eda_cities = CURATED_CITIES

    # ── Helper: dark styled matplotlib fig ───────────────────────────────────
    def dark_fig(w=10, h=4):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        return fig, ax

    # ═══════════════════════════════════════════════════════════════════════════
    if "Housing Price Change" in selected_var:
    # ═══════════════════════════════════════════════════════════════════════════
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 🏠 Housing Price Change (FHFA YoY %)")
            st.markdown("**In plain English:** Every year, we measure how much the average home price in a city went up or down compared to the year before. If it was +5%, homes got 5% more expensive.")
            st.markdown("**Where it comes from:** The Federal Housing Finance Agency (FHFA) — the official U.S. government source for home price data.")
            st.markdown("**Why it matters:** This is what we're trying to predict. Everything else in FirmScape is asking: *does this variable help explain why some cities' housing prices rose faster than others?*")
            st.caption("⚠️ This measures price *change*, not price *level*. A city with expensive homes and 0% growth looks the same as a cheap city with 0% growth.")

        with right:
            if use_idf:
                city_pick = st.selectbox("Pick a city to see its history:", eda_cities, key="eda_hpi_city")
                city_data = idf_eda[idf_eda['city_state'] == city_pick].dropna(subset=['fhfa_yoy']).sort_values(['year', 'quarter'])
                if not city_data.empty:
                    fig, ax = dark_fig()
                    ax.plot(range(len(city_data)), city_data['fhfa_yoy'].values, color='#4f8ef7', linewidth=2)
                    ax.axhline(0, color='#666', linewidth=0.8, linestyle='--')
                    ax.fill_between(range(len(city_data)), city_data['fhfa_yoy'].values, 0,
                                    where=city_data['fhfa_yoy'].values > 0, alpha=0.2, color='#4f8ef7')
                    ax.fill_between(range(len(city_data)), city_data['fhfa_yoy'].values, 0,
                                    where=city_data['fhfa_yoy'].values < 0, alpha=0.2, color='#f74f4f')
                    # X ticks at year boundaries
                    yr_rows = city_data.reset_index(drop=True)
                    ticks = yr_rows[yr_rows['quarter'] == 1].index[::4].tolist()
                    labels = yr_rows.loc[ticks, 'year'].astype(str).tolist()
                    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, color='white', fontsize=8)
                    ax.set_ylabel("% change vs year before", color='white')
                    ax.set_title(f"{city_pick} — Home Price Change Per Year (FHFA)", color='white')
                    st.pyplot(fig); plt.close(fig)
                    st.caption("🔵 Blue = prices rising. 🔴 Red = prices falling. The 2008 crash is visible in most cities.")

                    # ── Auto insight ──────────────────────────────────────────
                    st.markdown("#### 🧠 What does this tell us?")
                    hpi_vals = city_data['fhfa_yoy'].dropna()
                    avg = hpi_vals.mean()
                    volatility = hpi_vals.std()
                    max_yr = city_data.loc[city_data['fhfa_yoy'].idxmax(), 'year'] if not city_data.empty else "—"
                    min_yr = city_data.loc[city_data['fhfa_yoy'].idxmin(), 'year'] if not city_data.empty else "—"
                    pct_positive = (hpi_vals > 0).mean() * 100

                    col_h1, col_h2, col_h3 = st.columns(3)
                    col_h1.metric("Avg annual price change", f"{avg:.1f}%")
                    col_h2.metric("Volatility (std dev)", f"{volatility:.1f}%")
                    col_h3.metric("Years with rising prices", f"{pct_positive:.0f}%")
                    st.markdown(f"""
                    **For {city_pick}:** Home prices rose in **{pct_positive:.0f}% of quarters** on record, with an average change of **{avg:.1f}% per year**.
                    The biggest surge was around **{max_yr}** and the steepest drop around **{min_yr}**.
                    {'High volatility suggests this city is sensitive to economic shocks.' if volatility > 5 else 'Relatively stable price changes suggest a steadier local economy.'}
                    This is what the rest of FirmScape tries to explain — what drives those rises and falls?
                    """)
                else:
                    st.info("No FHFA data for this city.")

    # ═══════════════════════════════════════════════════════════════════════════
    elif "Firm Founding Rate" in selected_var:
    # ═══════════════════════════════════════════════════════════════════════════
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 🏗️ Firm Founding Rate (YoY %)")
            st.markdown("**In plain English:** How much did the number of new businesses in a city grow compared to last year? If a city had 100 new companies last year and 110 this year, that's +10%.")
            st.markdown("**Where it comes from:** Business registration records, aggregated by city and year.")
            st.markdown("**Why it matters:** New businesses mean new jobs, new workers moving in, more demand for housing. We expect this to *lead* housing prices — when firms boom, housing follows a year or two later.")
            st.caption("⚠️ A single year spike could be noise. Look for sustained growth over 3+ years for a real signal.")

        with right:
            if use_idf:
                city_pick2 = st.selectbox("Pick a city:", eda_cities, key="eda_firm_city")
                city_data2 = idf_eda[idf_eda['city_state'] == city_pick2].dropna(subset=['firms_founded_yoy', 'fhfa_yoy']).sort_values(['year', 'quarter'])
                if not city_data2.empty:
                    fig, ax = dark_fig()
                    ax2 = ax.twinx()
                    ax.plot(range(len(city_data2)), city_data2['firms_founded_yoy'].values,
                            color='#f7a44f', linewidth=2, label='Firm Growth %')
                    ax2.plot(range(len(city_data2)), city_data2['fhfa_yoy'].values,
                             color='#4f8ef7', linewidth=1.5, linestyle='--', alpha=0.7, label='Housing Price Change %')
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
                    ax.set_title(f"{city_pick2} — Firm Growth 🟠 vs Housing Prices 🔵", color='white')
                    lines1, labels_l1 = ax.get_legend_handles_labels()
                    lines2, labels_l2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels_l1 + labels_l2,
                              facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig); plt.close(fig)
                    st.caption("🟠 Orange = firm founding rate. 🔵 Blue dashed = housing price change. Does orange move before blue?")

                    # ── Auto-generated insight ────────────────────────────────
                    st.markdown("#### 🧠 What does this tell us?")
                    insight_df = city_data2[['firms_founded_yoy', 'fhfa_yoy']].dropna()

                    if len(insight_df) >= 8:
                        # Contemporaneous correlation
                        r_now, p_now = pearsonr(insight_df['firms_founded_yoy'], insight_df['fhfa_yoy'])

                        # Lag correlations: does firm growth LEAD housing by 1–4 quarters?
                        lag_results = {}
                        for lag in [1, 2, 4, 8]:
                            lagged = insight_df['firms_founded_yoy'].shift(lag)
                            combined = pd.DataFrame({'x': lagged, 'y': insight_df['fhfa_yoy']}).dropna()
                            if len(combined) >= 6 and combined['x'].std() > 0 and combined['y'].std() > 0:
                                r_lag, p_lag = pearsonr(combined['x'], combined['y'])
                                lag_results[lag] = (r_lag, p_lag)

                        best_lag = max(lag_results, key=lambda k: abs(lag_results[k][0])) if lag_results else None
                        best_r, best_p = lag_results[best_lag] if best_lag else (r_now, p_now)

                        # Plain-English summary
                        direction = "rise together" if r_now > 0.1 else ("move oppositely" if r_now < -0.1 else "don't move together much")
                        strength = "strongly" if abs(r_now) > 0.4 else ("moderately" if abs(r_now) > 0.2 else "weakly")
                        lag_quarters = best_lag or 0
                        lag_years = f"{lag_quarters // 4} year{'s' if lag_quarters > 4 else ''}" if lag_quarters >= 4 else f"{lag_quarters} quarter{'s' if lag_quarters > 1 else ''}"
                        lead_conclusion = (
                            f"The strongest signal comes when firm growth **leads housing by {lag_years}** "
                            f"(R = {best_r:.2f}). This {'supports' if abs(best_r) > abs(r_now) else 'does not clearly support'} "
                            f"the idea that firm growth is an early warning signal for housing prices in this city."
                        ) if best_lag else ""

                        sig_label = "statistically significant" if p_now < 0.05 else "not statistically significant"

                        col_i1, col_i2, col_i3 = st.columns(3)
                        col_i1.metric("Same-time correlation (R)", f"{r_now:.2f}")
                        col_i2.metric("Best lag", f"{lag_years}" if best_lag else "—")
                        col_i3.metric("Best lag R", f"{best_r:.2f}" if best_lag else "—")

                        st.markdown(f"""
                        **For {city_pick2}:** Firm founding growth and housing price changes {strength} {direction} at the same time (R = {r_now:.2f}, {sig_label}).

                        {lead_conclusion}

                        **Bottom line:** {'📈 Firm growth appears to be a useful leading indicator here — watch for sustained firm growth as an early signal of rising housing prices.' if abs(best_r) > 0.3 and best_lag else '⚠️ The relationship is weak for this city — other factors likely dominate housing prices here.'}
                        """)
                    else:
                        st.info("Not enough data points to run lag analysis for this city.")

    # ═══════════════════════════════════════════════════════════════════════════
    elif "Industry Concentration (HHI)" in selected_var:
    # ═══════════════════════════════════════════════════════════════════════════
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 🏭 Industry Concentration (HHI)")
            st.markdown("**In plain English:** Imagine a city where 90% of all jobs are at car factories. That city is *highly concentrated* — it's betting everything on one industry. The HHI score measures this. **High HHI = one industry dominates. Low HHI = many industries share the load.**")
            st.markdown("**Where it comes from:** Calculated from business registration data using the Herfindahl-Hirschman Index formula.")
            st.markdown("**Why it matters:** Highly concentrated cities are fragile — when Detroit's auto industry collapsed, the whole city collapsed. Diverse cities weather downturns better.")
            st.caption("⚠️ High concentration isn't always bad — Silicon Valley is concentrated in tech, and housing prices boomed. The *type* of industry matters as much as concentration.")

        with right:
            if use_idf:
                # Compare HHI over time for multiple cities
                comp_cities = st.multiselect("Compare cities (pick 2–4):", eda_cities, default=eda_cities[:3], key="eda_hhi_cities")
                if comp_cities:
                    fig, ax = dark_fig()
                    colors_hhi = ['#4f8ef7', '#f7a44f', '#4ff7a4', '#f74f4f']
                    for i, city in enumerate(comp_cities[:4]):
                        cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['hhi_new']).sort_values(['year', 'quarter'])
                        if not cd.empty:
                            ax.plot(range(len(cd)), cd['hhi_new'].values,
                                    color=colors_hhi[i], linewidth=2, label=city)
                    if comp_cities:
                        cd_ref = idf_eda[idf_eda['city_state'] == comp_cities[0]].dropna(subset=['hhi_new']).sort_values(['year', 'quarter'])
                        yr_rows3 = cd_ref.reset_index(drop=True)
                        ticks3 = yr_rows3[yr_rows3['quarter'] == 1].index[::4].tolist() if 'quarter' in yr_rows3.columns else []
                        labels3 = yr_rows3.loc[ticks3, 'year'].astype(str).tolist() if ticks3 else []
                        if ticks3: ax.set_xticks(ticks3); ax.set_xticklabels(labels3, rotation=45, color='white', fontsize=8)
                    ax.set_ylabel("HHI Score (higher = more concentrated)", color='white')
                    ax.set_title("Industry Concentration Over Time — by City", color='white')
                    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig); plt.close(fig)
                    st.caption("Higher line = fewer industries dominate the city's economy. Lower line = more spread out.")

    # ═══════════════════════════════════════════════════════════════════════════
    elif "Industry Diversity" in selected_var:
    # ═══════════════════════════════════════════════════════════════════════════
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 🌐 Industry Diversity")
            st.markdown("**In plain English:** How many *different types* of businesses exist in a city? A city with restaurants, hospitals, law firms, tech companies, and factories is *diverse*. A city with only steel mills is *not*.")
            st.markdown("**Where it comes from:** Count of distinct industry categories per city, per year.")
            st.markdown("**Why it matters:** More diverse = more stable. If one industry tanks, others can absorb the shock. We expect more diverse cities to have steadier (if not explosive) housing growth.")
            st.caption("⚠️ Diversity doesn't guarantee high prices — a city with lots of low-wage industries might be diverse but still see slow housing growth.")

        with right:
            if use_idf and 'industry_count_new' in idf_eda.columns:
                comp_cities2 = st.multiselect("Compare cities:", eda_cities, default=eda_cities[:3], key="eda_div_cities")
                if comp_cities2:
                    fig, ax = dark_fig()
                    colors_div = ['#4f8ef7', '#f7a44f', '#4ff7a4', '#f74f4f']
                    for i, city in enumerate(comp_cities2[:4]):
                        cd = idf_eda[idf_eda['city_state'] == city].dropna(subset=['industry_count_new']).sort_values(['year', 'quarter'])
                        if not cd.empty:
                            # Annual average to reduce noise
                            cd_annual = cd.groupby('year')['industry_count_new'].mean().reset_index()
                            ax.plot(cd_annual['year'], cd_annual['industry_count_new'],
                                    color=colors_div[i], linewidth=2, marker='o', markersize=3, label=city)
                    ax.set_xlabel("Year", color='white')
                    ax.set_ylabel("Number of distinct industries", color='white')
                    ax.set_title("Industry Diversity Over Time — by City", color='white')
                    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig); plt.close(fig)
                    st.caption("Each point = average number of distinct industries active in that city that year. Rising = getting more diverse.")

    # ═══════════════════════════════════════════════════════════════════════════
    elif "Top Industry Share" in selected_var:
    # ═══════════════════════════════════════════════════════════════════════════
        left, right = st.columns([1, 2])
        with left:
            st.markdown("#### 📊 Top Industry Share")
            st.markdown("**In plain English:** What percentage of all businesses in a city belong to the single biggest industry? If 40% of businesses are in healthcare, the top industry share is 40%.")
            st.markdown("**Where it comes from:** Calculated from business registration data — largest industry ÷ total businesses.")
            st.markdown("**Why it matters:** A high share means one industry is calling the shots for that city's economy — and its housing market. If that industry grows, housing booms. If it shrinks, housing suffers.")
            st.caption("⚠️ This is related to but different from HHI. HHI accounts for all industries; top share only looks at the #1.")

        with right:
            if use_idf and 'top_industry_share_new' in idf_eda.columns:
                comp_cities3 = st.multiselect("Compare cities:", eda_cities, default=eda_cities[:3], key="eda_top_cities")
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
                    ax.set_ylabel("Top industry's share of all businesses (%)", color='white')
                    ax.set_title("Top Industry Dominance Over Time — by City", color='white')
                    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig); plt.close(fig)
                    st.caption("Higher % = one industry dominates more. Watch how this changes before and after major economic events.")

    st.divider()

    # ── What our model actually does ─────────────────────────────────────────
    st.subheader("🤔 How Do We Use These Variables to Predict Housing Prices?")
    st.markdown("""
    Think of it like a recipe. Each variable above is an ingredient. Our model figures out 
    **which ingredients matter most** for predicting whether a city's housing prices will go up.

    Here's the idea:
    """)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("""
        **Step 1: Gather ingredients**
        - Firm founding rate
        - Industry concentration
        - Industry diversity
        - Top industry share
        """)
    with mc2:
        st.markdown("""
        **Step 2: Model learns weights**  
        The model asks: *if firm founding goes up 10%, how much does housing change?*  
        It finds the best answer from 50 years of real data.
        """)
    with mc3:
        st.markdown("""
        **Step 3: Be honest about limits**  
        Housing prices depend on *many* things we don't have — interest rates, zoning, supply.  
        A low R² score is **expected**, not a failure.
        """)

    st.info("💡 Go to **✅ Validation & Modeling** to run the actual models and see which variables come out on top.")

# ─────────────────────────────────────────────
# EVIDENCE TAB — powered by firmscape_integrated_cbsa_quarterly_cleaned
# ─────────────────────────────────────────────
if tab == "🔎 Evidence":
    st.title("What Patterns Show Up Across Cities?")
    st.markdown("Explore the relationship between industrial clustering and urban housing value.")

    # ── Check for integrated dataset ──────────────────────────────────────────
    if integrated_df is None:
        import glob
        found_files = glob.glob("*.csv")
        st.error("⚠️ Could not find the integrated dataset. Files found in current directory:")
        st.code("\n".join(sorted(found_files)) if found_files else "No CSV files found — is app.py in the same folder as your CSVs?")
        st.info("Looked for: firmscape_integrated_cbsa_quarterly_cleaned.csv and similar names. Rename your file to match or place it in the same folder as app.py.")
        st.stop()
    else:
        st.caption(f"✅ Loaded: `{integrated_fname}`")

    idf = integrated_df.copy()
    # Detect which column holds city names
    city_col_name = idf['_city_col'].iloc[0] if '_city_col' in idf.columns else 'city_state'
    idf['city_state'] = idf[city_col_name].astype(str)  # normalize to city_state

    # ── Curated cities from integrated dataset ────────────────────────────────
    # Let user pick from real cities in the dataset, defaulting to familiar ones
    all_int_cities = sorted(idf['city_state'].unique())
    PREFERRED = ["Austin, TX", "Detroit, MI", "San Jose, CA", "Seattle, WA", "Chicago, IL"]
    default_cities = [c for c in PREFERRED if c in all_int_cities] or all_int_cities[:5]

    # ── 1. CITY TIMELINE: Housing Price + Firm Growth Over Time ──────────────
    st.subheader("🏙️ City Timeline: Housing Price & Firm Growth")
    st.caption("Uses quarterly FHFA / Zillow housing data and firm founding rates from 1977–present.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        # Auto-select preset city if one was loaded from Home tab
        preset_city = st.session_state.get('preset_city')
        default_city_idx = 0
        city_options = default_cities + [c for c in all_int_cities if c not in default_cities]
        if preset_city and preset_city in city_options:
            default_city_idx = city_options.index(preset_city)
        elif preset_city:
            # Try partial match (e.g. "San Jose, CA" vs "San Jose, CA")
            for i, c in enumerate(city_options):
                if preset_city.split(",")[0].lower() in c.lower():
                    default_city_idx = i
                    break
        city_choice = st.selectbox("Select a City:", city_options, index=default_city_idx, key="ev_city")

    with col_c2:
        housing_options = [
            "fhfa_yoy (FHFA % change YoY)",
            "fhfa_index (FHFA Index Level)",
            "zillow_yoy (Zillow % change YoY)",
            "zillow_price_q (Zillow Price)"
        ]
        preset_metric = st.session_state.get('preset_housing_metric')
        default_metric_idx = 0
        if preset_metric and preset_metric in housing_options:
            default_metric_idx = housing_options.index(preset_metric)
        housing_metric = st.selectbox("Housing Metric:", housing_options, index=default_metric_idx, key="ev_housing")
        h_col = housing_metric.split(" ")[0]

    # Show preset banner if active
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
        # Build a clean x-axis from yq
        city_ts = city_ts.dropna(subset=[h_col, 'firms_founded_yoy'], how='all')

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=city_ts['yq'], y=city_ts[h_col],
            name=housing_metric.split("(")[1].rstrip(")"),
            line=dict(color='#4f8ef7', width=2),
            yaxis='y1'
        ))
        fig_ts.add_trace(go.Scatter(
            x=city_ts['yq'], y=city_ts['firms_founded_yoy'],
            name='Firm Founding YoY %',
            line=dict(color='#f7a44f', width=2, dash='dot'),
            yaxis='y2'
        ))
        fig_ts.update_layout(
            title=f"{city_choice} — Housing vs Firm Growth Over Time",
            template='plotly_dark',
            yaxis=dict(title=h_col, titlefont=dict(color='#4f8ef7')),
            yaxis2=dict(title='Firm Founding YoY %', titlefont=dict(color='#f7a44f'),
                        overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=400,
            xaxis=dict(
                tickmode='array',
                tickvals=city_ts['yq'].iloc[::16].tolist(),  # tick every 4 years (16 quarters)
                tickangle=45
            )
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption(
            "🔵 Housing metric (left axis) vs 🟠 Firm founding growth rate (right axis). "
            "Look for the orange line to lead the blue — that's the signal."
        )

    st.divider()

    # ── 2. SCATTER PLOT: Industry Concentration vs Housing Price Change ───────
    st.subheader("🔵 Scatter Plot: Industry Concentration vs Housing Changes")
    st.caption("Each dot = one city-quarter observation. Drag to zoom, hover for details.")

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        x_axis = st.selectbox("X-axis (Industry Variable):", [
            "hhi_new — Market Concentration (HHI)",
            "top_industry_share_new — Top Industry Share",
            "industry_count_new — # of Industries",
            "firm_count_total — Total Firms",
            "firms_founded_yoy — Firm Growth Rate YoY"
        ], key="sc_x")
        x_col = x_axis.split(" — ")[0]

    with sc_col2:
        y_axis = st.selectbox("Y-axis (Housing Variable):", [
            "fhfa_yoy — FHFA Housing % Change YoY",
            "fhfa_index — FHFA Index Level",
            "zillow_yoy — Zillow % Change YoY",
            "zillow_price_q — Zillow Price Level"
        ], key="sc_y")
        y_col = y_axis.split(" — ")[0]

    with sc_col3:
        color_by = st.selectbox("Color dots by:", [
            "year — Year",
            "city_state — City"
        ], key="sc_color")
        color_col = color_by.split(" — ")[0]

    # Year filter slider
    yr_range = st.slider(
        "Filter by Year Range:",
        min_value=int(idf['year'].min()),
        max_value=int(idf['year'].max()),
        value=(int(idf['year'].min()), int(idf['year'].max())),
        key="sc_yr"
    )

    sc_df = idf[
        (idf['year'] >= yr_range[0]) & (idf['year'] <= yr_range[1])
    ].dropna(subset=[x_col, y_col])

    # Cap outliers for cleaner scatter
    x_p1, x_p99 = sc_df[x_col].quantile([0.01, 0.99])
    y_p1, y_p99 = sc_df[y_col].quantile([0.01, 0.99])
    sc_df = sc_df[(sc_df[x_col].between(x_p1, x_p99)) & (sc_df[y_col].between(y_p1, y_p99))]

    if not sc_df.empty:
        plot_df = sc_df.sample(min(len(sc_df), 3000), random_state=42).copy()

        # ── Use matplotlib (no WebGL needed) ──────────────────────────────────
        fig_sc, ax_sc = plt.subplots(figsize=(10, 5))
        fig_sc.patch.set_facecolor("#0e1117")
        ax_sc.set_facecolor("#0e1117")

        if color_col == 'year':
            sc_scatter = ax_sc.scatter(
                plot_df[x_col], plot_df[y_col],
                c=plot_df['year'], cmap='viridis',
                alpha=0.45, s=18, linewidths=0
            )
            cbar = fig_sc.colorbar(sc_scatter, ax=ax_sc)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
            cbar.set_label('Year', color='white')
        else:
            # Color by city — pick top 10 cities by count, grey out rest
            top_cities = plot_df['city_state'].value_counts().head(10).index
            cmap_cities = plt.cm.get_cmap('tab10', 10)  # type: ignore
            for i, city in enumerate(top_cities):
                mask = plot_df['city_state'] == city
                ax_sc.scatter(plot_df.loc[mask, x_col], plot_df.loc[mask, y_col],
                              color=cmap_cities(i), alpha=0.6, s=18, linewidths=0,
                              label=city)
            # Grey out remaining
            other_mask = ~plot_df['city_state'].isin(top_cities)
            ax_sc.scatter(plot_df.loc[other_mask, x_col], plot_df.loc[other_mask, y_col],
                          color='#555', alpha=0.25, s=12, linewidths=0)
            ax_sc.legend(fontsize=7, facecolor='#1e1e2e', labelcolor='white',
                         markerscale=1.5, loc='upper right')

        # OLS trendline using numpy (no WebGL)
        sc_clean_plot = plot_df[[x_col, y_col]].dropna()
        if len(sc_clean_plot) >= 5:
            m, b = np.polyfit(sc_clean_plot[x_col], sc_clean_plot[y_col], 1)
            x_line = np.linspace(sc_clean_plot[x_col].min(), sc_clean_plot[x_col].max(), 200)
            ax_sc.plot(x_line, m * x_line + b, color='white', linewidth=1.5,
                       linestyle='--', label='OLS trendline', zorder=5)

        ax_sc.set_xlabel(x_axis.split(' — ')[1], color='white', fontsize=10)
        ax_sc.set_ylabel(y_axis.split(' — ')[1], color='white', fontsize=10)
        ax_sc.set_title(
            f"{x_axis.split(' — ')[1]}  vs  {y_axis.split(' — ')[1]}",
            color='white', fontsize=12
        )
        ax_sc.tick_params(colors='white')
        ax_sc.spines[:].set_color('#333')
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        # Correlation stats below chart
        sc_clean = sc_df[[x_col, y_col]].dropna()
        if len(sc_clean) >= 5 and sc_clean[x_col].std() > 0 and sc_clean[y_col].std() > 0:
            r_sc, p_sc = pearsonr(sc_clean[x_col], sc_clean[y_col])
            r2_sc = r_sc ** 2
            sig = "✅ Significant" if p_sc < 0.05 else "❌ Not significant"
            sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
            sc_c1.metric("Pearson R", f"{r_sc:.3f}")
            sc_c2.metric("R²", f"{r2_sc:.3f}")
            sc_c3.metric("P-Value", f"{p_sc:.4f}")
            sc_c4.metric("N (points)", f"{len(sc_clean):,}")
            st.caption(
                f"{sig} (p {'<' if p_sc < 0.05 else '≥'} 0.05). "
                f"R² = {r2_sc:.3f} means {x_axis.split(' — ')[1]} explains ~{r2_sc*100:.1f}% "
                f"of variance in {y_axis.split(' — ')[1]}. Low R² is expected — many factors drive housing prices."
            )
    else:
        st.warning("No data available for this combination. Adjust the year range or axes.")

    st.divider()

    # ── 3. MULTI-CITY HOUSING TREND (time series, up to 5 cities) ────────────
    st.subheader("📈 Compare Housing Trends Across Cities")

    mc_cities = st.multiselect(
        "Select cities to compare (up to 5):",
        all_int_cities,
        default=default_cities[:3],
        key="mc_cities"
    )
    mc_metric = st.selectbox(
        "Housing metric to compare:",
        ["fhfa_yoy", "fhfa_index", "zillow_yoy", "zillow_price_q"],
        key="mc_metric"
    )

    if mc_cities:
        mc_df = idf[idf['city_state'].isin(mc_cities[:5])].dropna(subset=[mc_metric])
        mc_df = mc_df.sort_values(['city_state', 'year', 'quarter'])

        fig_mc = px.line(
            mc_df, x='yq', y=mc_metric,
            color='city_state',
            labels={'yq': 'Quarter', mc_metric: mc_metric, 'city_state': 'City'},
            template='plotly_dark',
            title=f"{mc_metric} — Multi-City Comparison",
            height=400
        )
        # Only show every 4th year on x axis
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
            # Find quarter with max acceleration in firm growth
            city_ts_inf = city_ts_inf.copy()
            city_ts_inf['firm_accel'] = city_ts_inf['firms_founded_yoy'].diff()
            idx_max = city_ts_inf['firm_accel'].idxmax()
            inf_row = city_ts_inf.loc[idx_max]
            inf_yq = inf_row['yq']
            inf_year = int(inf_row['year'])

            growth_before = city_ts_inf[city_ts_inf['year'] < inf_year]['firms_founded_yoy'].mean()
            growth_after = city_ts_inf[city_ts_inf['year'] >= inf_year]['firms_founded_yoy'].mean()

            # Post-inflection housing correlation
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
            st.caption(f"P-value after inflection: {p_label}. R² measures how much firm growth explains housing price changes after the inflection.")

            # Chart: firm growth over time with inflection marked
            fig_inf, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(range(len(city_ts_inf)), city_ts_inf['firms_founded_yoy'].values,
                    color='#f7a44f', linewidth=2, label='Firm Growth YoY %')
            ax.plot(range(len(city_ts_inf)), city_ts_inf['fhfa_yoy'].values,
                    color='#4f8ef7', linewidth=2, label='FHFA YoY %', alpha=0.8)

            # Mark inflection
            inf_pos = city_ts_inf.index.get_loc(idx_max)
            ax.axvline(inf_pos, color='yellow', linestyle='--', linewidth=1.5, label=f'Inflection ({inf_yq})')
            ax.axvspan(inf_pos, min(inf_pos + 8, len(city_ts_inf) - 1),
                       color='yellow', alpha=0.12, label='Post-Inflection Window')

            # X ticks at years
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
# VALIDATION & MODELING TAB (revised)
# ─────────────────────────────────────────────
if tab == "✅ Validation & Modeling":
    st.title("✅ Validation & Interactive Modeling")
    st.markdown(
        "Test how well industry variables predict housing price changes. "
        "**A low R² is expected** — housing prices have many drivers. "
        "Our goal is to find *which variables matter most*, not to overfit."
    )

    # --- MODEL SELECTION ---
    st.subheader("Step 1: Choose a Model")
    model_choice = st.selectbox(
        "Select a model to run:",
        ["Linear Regression (Interpretable)", "XGBoost (Best Performance)", "Random Forest (Feature Importance)"],
        key="model_selector"
    )

    # --- HYPOTHESIS SELECTION ---
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

    # --- LAG TESTING ---
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

    # --- PRACTICAL SIGNIFICANCE ---
    st.divider()
    st.subheader("Step 4: Practical Significance")
    if not clean_val.empty and (not sig_filter or p_val < 0.05):
        st.info(f"""
        **The 'So What?' Factor:** A **10% increase** in company growth is associated with a 
        **{10 * slope:.2f}%** change in housing growth **{lag_years} year(s)** later.

        **R² = {r2:.3f}** — Industry concentration explains roughly **{r2*100:.1f}%** of housing 
        price variance. The remaining {(1-r2)*100:.1f}% is driven by other factors (interest rates, 
        supply, zoning, demographics). This is expected and does not invalidate the model.

        *(Note: This indicates association, not direct causation)*
        """)

    # --- INTERACTIVE MODEL RUN ---
    st.divider()
    st.subheader("Step 5: Run the Model Interactively")

    if st.button("🚀 Run Model"):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.preprocessing import LabelEncoder

        model_df = clean_val.copy()

        # Feature engineering
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
        )
        model_df = model_df.merge(div_df, on=["city_x", "year_int"], how="left")
        model_df['diversity'] = model_df['diversity'].fillna(1)

        features = ['pct_change', 'industry_enc', 'diversity']
        target = 'lagged_housing'

        X = model_df[features].dropna()
        y = model_df.loc[X.index, target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if "XGBoost" in model_choice:
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
                model_label = "XGBoost"
            except ImportError:
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
        rc1.metric("R² Score", f"{r2:.4f}", help="Fraction of variance explained. Low R² is expected — industry is one of many factors.")
        rc2.metric("RMSE", f"{rmse:.4f}")

        st.caption(
            f"💡 R² = {r2:.3f} means industry variables explain ~{r2*100:.1f}% of housing price variance. "
            "This is honest — not a weakness. There are many confounding variables. "
            "The value is in knowing *which variables* are most predictive."
        )

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            fi = pd.DataFrame({
                "Feature": ["Company Growth %", "Industry Type", "Industry Diversity"],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                template="plotly_dark",
                title="Feature Importance — Which Variables Drive Housing Price Changes?"
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption(
                "📌 Feature importance shows which variables contribute most to the model's predictions. "
                "Use this alongside the EDA Explorer to understand *why* a variable matters."
            )
        elif model_label == "Linear Regression":
            coef_df = pd.DataFrame({
                "Feature": ["Company Growth %", "Industry Type", "Industry Diversity"],
                "Coefficient": model.coef_
            }).sort_values("Coefficient", key=abs, ascending=True)
            fig_coef = px.bar(
                coef_df, x="Coefficient", y="Feature", orientation="h",
                color="Coefficient", color_continuous_scale="RdBu",
                template="plotly_dark",
                title="Linear Regression Coefficients"
            )
            st.plotly_chart(fig_coef, use_container_width=True)

    # --- COMPARE CITIES (curated, professor said 3-5 familiar cities) ---
    st.divider()
    st.subheader("Step 6: Compare Curated Cities")
    st.caption("We compare a curated set of economically distinct cities to avoid overwhelming comparisons.")

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
                    # Guard: pearsonr is undefined if either array is constant
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
            st.caption(
                "📌 R² values are expected to be low — this means industry data alone doesn't fully predict housing prices. "
                "Significant p-values confirm the relationship exists even if partial."
            )

# ─────────────────────────────────────────────
# OPPORTUNITY LAB TAB
# ─────────────────────────────────────────────
if tab == "🚀 Opportunity Lab":
    st.title("🚀 Opportunity Lab: The 'Next Hub' Finder")
    st.markdown(
        "Build your own investment shortlist by weighting the economic signals that matter most to you. "
        "Uses the **firmscape_integrated_quarterly** dataset when available."
    )

    # Use integrated dataset if available, else fallback
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

    # Strategy weights
    with st.sidebar:
        st.header("⚖️ Strategy Weights")
        w_growth = st.slider("Company Growth", 0.0, 1.0, 0.4, key="w_growth")
        w_diversity = st.slider("Industry Diversity", 0.0, 1.0, 0.3, key="w_div")
        w_afford = st.slider("Housing Affordability", 0.0, 1.0, 0.2, key="w_aff")
        w_stability = st.slider("Price Stability", 0.0, 1.0, 0.1, key="w_stab")
        st.divider()
        lookback = st.slider("Analysis Window (Years)", 1, 5, 3, key="lookback")

        st.caption(
            "💡 These weights reflect *your priorities* as a "
            + ("housing investor." if stakeholder == "🏠 Housing Investor"
               else "business analyst." if stakeholder == "📊 Business Analyst"
               else "researcher.")
        )

    # Score computation
    opp_df = opp_df_source.copy()

    if div_col and div_col in opp_df.columns:
        city_stats = opp_df.groupby(city_col).agg(
            growth=(growth_col, 'mean'),
            diversity=(div_col, 'mean')
        ).reset_index()
    else:
        city_stats = opp_df.groupby(city_col).agg(
            growth=(growth_col, 'mean')
        ).reset_index()
        city_stats['diversity'] = 1  # fallback

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

    # Leaderboard
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

    st.caption(
        "📌 Scores reflect a weighted combination of company growth and industry diversity. "
        "Adjust the sliders to match your investment priorities. "
        "This is a screening tool — always validate with additional data sources."
    )

    # What-If Shock Simulator
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
                st.caption(
                    "Note: Shock simulations apply a simple multiplier to illustrate directional impact. "
                    "Real impacts depend on many market factors."
                )

    # Judge's Shortlist
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
                st.write(f"⚠️ **Risk:** Industry data partially predicts housing — verify with macro factors.")

        st.caption(
            "💡 These cities score highest given your current weight settings. "
            "Low R² in the model is a reminder: use this as a starting point for deeper due diligence, "
            "not a definitive prediction."
        )
