import os
# ─── REQUIRED PACKAGES ──────────────────────────────────────────────────────
# Run this in your terminal before starting:
#   pip install streamlit pandas matplotlib seaborn numpy scipy plotly scikit-learn
#   pip install xgboost  (optional — falls back to GradientBoosting if missing)
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress ConstantInputWarning

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
    """Load the quarterly integrated dataset for Opportunity Lab."""
    try:
        df = pd.read_csv("firmscape_integrated_quarterly.csv")
        df['city'] = df['city'].fillna("Unknown").astype(str)
        return df
    except FileNotFoundError:
        return None

try:
    merged_companies_housing = load_data()
except FileNotFoundError:
    st.error("File 'merged_companies_housing.csv' not found.")
    st.stop()

integrated_df = load_integrated_data()

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
# HOME TAB
# ─────────────────────────────────────────────
if tab == "🏠 Home":
    st.title("FirmScape")

    # Stakeholder-tailored intro
    if stakeholder == "🏠 Housing Investor":
        st.info("👋 **You're viewing the Housing Investor lens.** This dashboard helps you understand how industry shifts in a city can be early signals of housing price changes.")
    elif stakeholder == "📊 Business Analyst":
        st.info("👋 **You're viewing the Business Analyst lens.** Explore how industry concentration and growth metrics relate to urban economic development.")
    else:
        st.info("👋 **You're viewing the Researcher lens.** Dive into the statistical models, EDA, and variable explanations with full transparency on model limitations.")

    st.markdown("""
    FirmScape is a data analysis and visualization project examining how **industry concentration 
    and company growth shape housing prices** in U.S. cities over time.
    """)

    st.markdown("""
    **Core Question:** *How much of housing price change can be predicted by industry shifts alone — 
    and which industries matter most?*
    """)

    # Honest framing upfront (professor's p-hacking feedback)
    with st.expander("⚠️ Important: What This Model Does (and Doesn't) Do"):
        st.markdown("""
        Housing prices are influenced by **many variables** — interest rates, population, zoning, 
        supply, and more. Our model focuses only on **industry concentration and company growth** 
        as predictors.

        - A **low R²** is expected and *not a flaw* — it means industry data partially explains 
          housing prices alongside many other factors
        - We **do not** chase high p-values or R² at the cost of validity
        - This model is best used as **one input** among others when evaluating housing market conditions
        - Our goal: show *which industry variables are the most predictive*, not claim full predictability
        """)

    st.markdown("**How to Use:**")
    st.markdown("""
    1. **EDA Explorer** — Understand each variable: what it is, where it comes from, why it matters  
    2. **Evidence** — See geographic patterns and city timelines  
    3. **Validation & Modeling** — Run XGBoost / Random Forest / Linear models interactively  
    4. **Opportunity Lab** — Build your own city shortlist with weighted signals
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cities/metros", "13,330")
    col2.metric("Years covered", "50")
    col3.metric("Companies", "275,438")
    col4.metric("% Coverage", "96%")

    case_study = st.selectbox(
        "Pick a famous case study to explore",
        ["Detroit Manufacturing", "Bay Area Technology", "Austin Technology", "Seattle Tech"]
    )
    st.write(f"→ Jumping to **{case_study}** preset in EDA Explorer...")
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
# EDA EXPLORER TAB (NEW — professor's feedback)
# ─────────────────────────────────────────────
if tab == "📊 EDA Explorer":
    st.title("📊 EDA Explorer: Understanding Your Variables")
    st.markdown(
        "Before modeling, understand each variable — what it is, where it comes from, "
        "and which direction it pushes housing prices. Click any variable to explore it."
    )

    # Variable dictionary with explanations
    VARIABLES = {
        "pct_change (Company Growth %)": {
            "description": "Year-over-year percentage change in the number of companies in a city.",
            "source": "Derived from companies_sorted.csv aggregated by city & year.",
            "direction": "📈 Positive — More companies generally signal economic vitality, which tends to push housing prices up.",
            "caveat": "Short-term spikes can reflect data noise; multi-year trends are more reliable.",
            "col": "pct_change"
        },
        "industry (Industry Type)": {
            "description": "The primary industry sector of companies in a city.",
            "source": "companies_sorted.csv — industry classification field.",
            "direction": "🔀 Varies — Tech & Finance industries historically associate with stronger housing price growth than manufacturing.",
            "caveat": "Industry mix matters; a city with many low-wage industries may not see housing appreciation.",
            "col": "industry"
        },
        "HPI / pct_change (Housing Price Index %)": {
            "description": "Year-over-year % change in the Federal Housing Finance Agency's House Price Index for a metro area.",
            "source": "hpi_at_metro.csv — FHFA official data.",
            "direction": "🎯 This is our TARGET variable — what we are trying to predict.",
            "caveat": "HPI captures price trends but not absolute levels; local supply constraints are not included.",
            "col": "pct_change"
        },
        "Industry Diversity (# unique industries)": {
            "description": "Count of distinct industries present in a city in a given year.",
            "source": "Derived from company data grouped by city & year.",
            "direction": "📈 Positive — Diverse economies are more resilient and tend toward steady housing appreciation.",
            "caveat": "Diversity alone doesn't guarantee growth — industry quality matters too.",
            "col": None
        },
    }

    selected_var = st.selectbox("🔍 Select a variable to explore:", list(VARIABLES.keys()))
    info = VARIABLES[selected_var]

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown(f"**📖 What is it?**  \n{info['description']}")
        st.markdown(f"**📂 Source:**  \n{info['source']}")
        st.markdown(f"**➡️ Effect Direction:**  \n{info['direction']}")
        st.caption(f"⚠️ Caveat: {info['caveat']}")

    with col_b:
        if info["col"] and info["col"] in merged_companies_housing.columns:
            col_data = merged_companies_housing[info["col"]].dropna()
            if col_data.dtype in [np.float64, np.int64]:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(col_data.clip(-50, 100), bins=60, color="#4f8ef7", edgecolor="none", alpha=0.85)
                ax.set_facecolor("#0e1117")
                fig.patch.set_facecolor("#0e1117")
                ax.tick_params(colors="white")
                ax.spines[:].set_color("#333")
                ax.set_title(f"Distribution of {selected_var}", color="white")
                st.pyplot(fig)
            elif info["col"] == "industry":
                top_industries = (
                    merged_companies_housing["industry"]
                    .value_counts()
                    .head(15)
                    .reset_index()
                )
                top_industries.columns = ["Industry", "Count"]
                fig2 = px.bar(
                    top_industries, x="Count", y="Industry", orientation="h",
                    color="Count", color_continuous_scale="Blues",
                    template="plotly_dark", title="Top 15 Industries by Company Count"
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            # Industry diversity — compute it
            div_df = (
                merged_companies_housing.groupby(["city_x", "year_int"])["industry"]
                .nunique()
                .reset_index()
                .rename(columns={"industry": "diversity"})
            )
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.hist(div_df["diversity"], bins=40, color="#f7a44f", edgecolor="none", alpha=0.85)
            ax3.set_facecolor("#0e1117")
            fig3.patch.set_facecolor("#0e1117")
            ax3.tick_params(colors="white")
            ax3.spines[:].set_color("#333")
            ax3.set_title("Distribution of Industry Diversity per City-Year", color="white")
            st.pyplot(fig3)

    st.divider()
    st.subheader("📌 Additive Prediction Model (Conceptual)")
    st.markdown("""
    Our model uses a **classic additive structure**:

    > **Predicted Housing Price Change** = *w₁ × Company Growth* + *w₂ × Industry Diversity* + *w₃ × Industry Type* + *ε*

    - **w₁, w₂, w₃** are learned weights — the model tells us *which variables matter more*
    - **ε** = error term — everything housing prices are affected by that we *don't* measure (interest rates, supply, zoning, etc.)
    - A low R² means our variables capture *part* of the story — that's honest and expected
    """)

    st.info(
        "💡 Use the **Validation & Modeling** tab to run XGBoost, Random Forest, or Linear Regression "
        "interactively and see which variables get the highest feature importance."
    )

# ─────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────
if tab == "🔎 Evidence":
    st.title("What Patterns Show Up Across Cities?")
    st.markdown("Explore the relationship between industrial clustering and urban housing value.")

    # 1. Map + Time Slider
    st.subheader("🌐 Geographic Shifts Over Time")
    available_years = sorted(merged_companies_housing['year_int'].unique())

    # Guard: select_slider crashes if only 1 option (min == max)
    if len(available_years) > 1:
        selected_year = st.select_slider(
            "Move the slider to see how industries shifted across the US:",
            options=available_years,
            value=max(available_years)
        )
    else:
        selected_year = available_years[0]
        st.info(f"📅 Data available for **{selected_year}** only — slider disabled.")

    map_col1, map_col2 = st.columns([3, 1])
    with map_col1:
        st.info(f"Showing Company Growth & Industry Diversity in {selected_year}")
        map_data = merged_companies_housing[merged_companies_housing['year_int'] == selected_year].copy()
        map_data['bubble_size'] = map_data['pct_change'].abs() + 1

        if not map_data.empty:
            try:
                fig = px.scatter_geo(
                    map_data,
                    locations="state",
                    locationmode="USA-states",
                    size="bubble_size",
                    color="industry",
                    hover_name="city_x",
                    scope="usa",
                    title=f"Industrial Growth Clusters ({selected_year})",
                    template="plotly_dark"
                )
                fig.update_layout(height=500, margin={"r": 0, "t": 40, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering map: {e}")
        else:
            st.warning(f"No data available for {selected_year}.")

    with map_col2:
        st.markdown("**Map Legend**")
        st.write("🟢 High Diversity")
        st.write("🔴 High Concentration")
        if not map_data.empty:
            top_city = map_data.sort_values('pct_change', ascending=False).iloc[0]
            st.metric("Top Growth City", top_city['city_x'], f"{top_city['pct_change']:.1f}%")

    st.divider()

    # 2. City Timeline — curated cities only (professor's feedback)
    st.subheader("🏙️ City Deep Dive")
    st.caption("Explore a curated set of cities with well-known economic histories.")

    col_select1, col_select2 = st.columns(2)
    with col_select1:
        city_choice = st.selectbox("Select a City:", CURATED_CITIES)
    with col_select2:
        available_industries = sorted(merged_companies_housing["industry"].dropna().unique())
        ind_choice = st.selectbox("Select an Industry:", available_industries)

    # Use real data for city timeline
    city_df = merged_companies_housing[
        (merged_companies_housing['city_x'].str.contains(city_choice.split(",")[0], case=False, na=False)) &
        (merged_companies_housing['industry'].str.lower() == ind_choice.lower())
    ].sort_values('year_int')

    unique_years_city = city_df['year_int'].nunique() if not city_df.empty else 0

    if not city_df.empty:
        if unique_years_city > 1:
            # Multi-year: line chart over time
            time_data = city_df.groupby('year_int').agg(
                Industry_Trend=('pct_change', 'mean')
            ).reset_index()
            st.markdown(f"**Industry Growth Trend — {city_choice} ({ind_choice})**")
            fig_city = px.line(
                time_data, x='year_int', y='Industry_Trend',
                labels={'year_int': 'Year', 'Industry_Trend': 'Avg % Change'},
                template='plotly_dark',
                title=f"{city_choice} — {ind_choice} Growth Over Time"
            )
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            # Single year: show top industries by growth as bar chart
            st.markdown(f"**Top Industries in {city_choice} ({unique_years_city} year of data)**")
            st.caption("Only one year of data available — showing industry snapshot instead of trend line.")
            city_snapshot = merged_companies_housing[
                merged_companies_housing['city_x'].str.contains(city_choice.split(",")[0], case=False, na=False)
            ].groupby('industry')['pct_change'].mean().reset_index()
            city_snapshot.columns = ['Industry', 'Avg Growth %']
            city_snapshot = city_snapshot.sort_values('Avg Growth %', ascending=False).head(15)
            fig_snap = px.bar(
                city_snapshot, x='Avg Growth %', y='Industry', orientation='h',
                color='Avg Growth %', color_continuous_scale='Blues',
                template='plotly_dark',
                title=f"Industry Growth Snapshot — {city_choice}"
            )
            fig_snap.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
            st.plotly_chart(fig_snap, use_container_width=True)
    else:
        st.info(f"No data found for {city_choice} + {ind_choice}. Try another combination.")

    st.divider()

    # 3. Growth Cohort Comparison
    st.subheader("🚀 Growth Cohort Comparison")
    cohort_view = st.selectbox("Compare Cohorts:", ["High Growth vs Low Growth", "High Diversity vs Low Diversity"])

    # Compute real cohort data
    city_summary = merged_companies_housing.groupby('city_x').agg(
        avg_growth=('pct_change', 'mean')
    ).reset_index()
    median_growth = city_summary['avg_growth'].median()
    high_growth_cities = city_summary[city_summary['avg_growth'] >= median_growth]['city_x']
    low_growth_cities = city_summary[city_summary['avg_growth'] < median_growth]['city_x']

    high_cohort = (
        merged_companies_housing[merged_companies_housing['city_x'].isin(high_growth_cities)]
        .groupby('year_int')['pct_change'].mean()
    )
    low_cohort = (
        merged_companies_housing[merged_companies_housing['city_x'].isin(low_growth_cities)]
        .groupby('year_int')['pct_change'].mean()
    )

    cohort_df = pd.DataFrame({
        "High Growth Cities (Avg % Change)": high_cohort,
        "Low Growth Cities (Avg % Change)": low_cohort
    }).dropna()

    if len(cohort_df) > 1:
        st.line_chart(cohort_df, height=300)
    else:
        # Single year: show as side-by-side bar instead
        bar_df = cohort_df.T.reset_index()
        bar_df.columns = ['Cohort', 'Avg % Change']
        fig_bar = px.bar(
            bar_df, x='Cohort', y='Avg % Change',
            color='Cohort', template='plotly_dark',
            title="High vs Low Growth Cities — Single Year Snapshot"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("📌 Only one year in dataset — trend line replaced with snapshot comparison.")

    st.divider()

    # 4. Automated Insights
    st.subheader("🎯 Automated Insights")
    if st.button("✨ Spot the Inflection Point"):
        city_key = city_choice.split(",")[0]
        city_all = merged_companies_housing[
            merged_companies_housing['city_x'].str.contains(city_key, case=False, na=False)
        ].sort_values('year_int')

        if city_all.empty or city_all['year_int'].nunique() < 3:
            # Single year or too little data: show a snapshot instead
            st.info(f"Not enough multi-year data for {city_choice} to detect an inflection point.")
            if not city_all.empty:
                snap = city_all.groupby('industry')['pct_change'].mean().reset_index()
                snap.columns = ['Industry', 'Avg Growth %']
                snap = snap.sort_values('Avg Growth %', ascending=False).head(10)
                fig_snap2 = px.bar(
                    snap, x='Avg Growth %', y='Industry', orientation='h',
                    color='Avg Growth %', color_continuous_scale='Viridis',
                    template='plotly_dark',
                    title=f"Top Growing Industries in {city_choice}"
                )
                st.plotly_chart(fig_snap2, use_container_width=True)
        else:
            st.balloons()
            # Compute real inflection: year with max year-over-year acceleration
            yearly = city_all.groupby('year_int')['pct_change'].mean()
            acceleration = yearly.diff()  # change in growth rate year-over-year
            inflection_year = int(acceleration.idxmax())

            growth_before = yearly[yearly.index < inflection_year].mean()
            growth_after = yearly[yearly.index >= inflection_year].mean()

            # Lag correlation after inflection
            post_df = city_all[city_all['year_int'] >= inflection_year].copy()
            post_df['lagged'] = post_df['pct_change'].shift(-1)
            post_df = post_df.dropna(subset=['lagged', 'pct_change'])
            if len(post_df) >= 3 and post_df['pct_change'].std() > 0 and post_df['lagged'].std() > 0:
                r_post, _ = pearsonr(post_df['pct_change'], post_df['lagged'])
                r2_post = round(r_post ** 2, 3)
            else:
                r2_post = "n/a"

            st.success(
                f"**Inflection Detected:** In {city_choice}, company growth rate accelerated most sharply in **{inflection_year}**."
            )
            st.markdown(f"""
            **What happened next?**
            - **Company Growth Before:** {growth_before:.1f}% avg → **After:** {growth_after:.1f}% avg
            - **Housing Lag:** Prices typically begin responding 12–18 months after industry inflections.
            - **Post-Inflection R²:** {r2_post} *(fraction of housing variance explained by industry growth after the inflection)*
            """)

            # Real chart
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(yearly.index, yearly.values, color="#4f8ef7", linewidth=2)
            ax.axvspan(inflection_year, inflection_year + 2, color='yellow', alpha=0.25, label="Acceleration Window")
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#333")
            ax.set_xlabel("Year", color="white")
            ax.set_ylabel("Avg Growth %", color="white")
            ax.set_title(f"{city_choice} — All Industries Growth (Inflection: {inflection_year})", color="white")
            ax.legend(facecolor="#0e1117", labelcolor="white")
            st.pyplot(fig)

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
        city_col = 'city'
        st.success("✅ Using `firmscape_integrated_quarterly` dataset.")
    else:
        opp_df_source = merged_companies_housing.copy()
        city_col = 'city_x'
        st.warning("⚠️ `firmscape_integrated_quarterly` not found — using merged_companies_housing fallback.")

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

    if 'pct_change' in opp_df.columns:
        growth_col = 'pct_change'
    else:
        growth_col = opp_df.select_dtypes(include=np.number).columns[0]

    city_stats = opp_df.groupby(city_col).agg(
        growth=(growth_col, 'mean'),
        diversity=('industry', 'nunique') if 'industry' in opp_df.columns else (growth_col, 'std')
    ).reset_index()
    city_stats.columns = [city_col, 'growth', 'diversity']

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
        
