import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# Sidebar navigation
st.sidebar.title("FirmScape Dashboard")
tab = st.sidebar.radio(
    "Select a tab",
    ["🏠 Home", "🧩 Build the Dataset", "🔎 Evidence", "✅ Validation", "🚀 Opportunity Lab"]
)

if tab == "🏠 Home":
    st.title("FirmScape")
    st.markdown("""
    FirmScape is a multi-scape data analysis and visualization project that examines how industry 
    concentration and company growth shape the rise and decline of U.S. cities over time. 
    By analyzing historial company and housing data, we aim to understand the dynamics of urban development 
    and identify key factors influencing city growth and decline. )
                """)
    st.markdown("""
    **Core Question:** When industries cluster and grow in a city, how do housing prices move—now and later?
    """)
    st.markdown("""
    **How to Use:**  
    1. Explore city + industry patterns over time  
    2. Validate relationships with lag + significance  
    3. Compare cities and generate a 'Next Hub' shortlist
    """)
    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cities/metros", 13330)
    col2.metric("Years covered", 50)
    col3.metric("Companies", 275438)
    col4.metric("% Coverage", "96%")
    
    # Case study dropdown
    case_study = st.selectbox(
        "Pick a famous case study", 
        ["Detroit Manufacturing", "Bay Area Technology", "Austin Technology", "Seattle Tech"]
    )
    st.write(f"Jumping to {case_study} preset for Tab 2...")
    st.markdown("*Disclaimer: No casual claims — all data-driven.*")

if tab == "🧩 Build the Dataset":
    st.title("From Messy Sources to one City Timeline")
 
    # Pipeline diagram (can also use st.image if you have a graphic)
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

    "Rows": [7173426, 40258, 
             19486334, 0,
             83230, 69828,
             895, 230406,
             275438],

    "Columns": [11, 7, 
                11, 0,
                6, 6,
                317, 7,
                15],
    "Notes": [
        "Some missing values",
        "Filtered to US companies with 100+ employees and dropped unwanted columns",
        "Very large dataset", 
        "Filtered to US companies with 100+ employees and dropped unwanted columns",
        "Needs to be recomputed for percent change",
        "Split location into city/state, created datetime column, & dropped rows without HPI and sort BEFORE pct_change",
        "Many empty rows",
        "Sort the data for time, fill small gaps, & dropped the NaNs",
        "Cleaned & Merged!"
    ]
    })
    st.dataframe(pipeline_data.head(9), height=500, width=900)
if tab == "🔎 Evidence":
    st.title("What patterns show up across cities?")
    st.markdown("Explore the relationship between industrial clustering and urban housing value.")

    # 1. Map + Time Slider Section
    st.subheader("🌐 Geographic Shifts Over Time")
    
    # Extract year from your 'date' column for the slider
    # We use your 'merged_companies_housing' dataframe here
    merged_companies_housing['year_int'] = pd.to_datetime(merged_companies_housing['date']).dt.year
    available_years = sorted(merged_companies_housing['year_int'].unique())
    
    selected_year = st.select_slider(
        "Move the slider to see how industries shifted across the US:",
        options=available_years,
        value=max(available_years)
    )

    map_col1, map_col2 = st.columns([3, 1])
    
    with map_col1:
        st.info(f"Showing Company Growth & Industry Diversity in {selected_year}")
        
        # Filter your actual data by the year selected on the slider
        map_data = merged_companies_housing[merged_companies_housing['year_int'] == selected_year].copy()

        # Create a bubble size column that handles negative pct_change values
        # Plotly size cannot be negative or zero
        map_data['bubble_size'] = map_data['pct_change'].abs() + 1 

        try:
            import plotly.express as px
            fig = px.scatter_geo(
                map_data,
                locations="state",
                locationmode="USA-states",
                size="bubble_size",      # Use the absolute value column for bubble size
                color="industry",        # Color bubbles by industry type
                hover_name="city_x",     # Show city name when hovering
                scope="usa",
                title=f"Industrial Growth Clusters ({selected_year})",
                template="plotly_dark"
            )
            
            fig.update_layout(height=500, margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering map: {e}")
            
    with map_col2:
        # Keep your existing metrics/legend here
        st.markdown("**Map Legend**")
        st.write("🟢 High Diversity")
        st.write("🔴 High Concentration")
        # You can make this metric dynamic too:
        top_city = map_data.sort_values('pct_change', ascending=False).iloc[0]
        st.metric("Top Growth City", top_city['city_x'], f"{top_city['pct_change']:.1f}%")

    st.divider()

    # 2. City Timeline Panel (Synchronized Charts)
    st.subheader("🏙️ City Timeline Panel")
    
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        city_choice = st.selectbox("Select a City:", ["Detroit, MI", "Austin, TX", "San Jose, CA", "Seattle, WA"])
    with col_select2:
        ind_choice = st.selectbox("Select an Industry:", ["Technology", "Manufacturing", "Finance", "Healthcare"])

    # Synchronized Line Charts
    # Mock data for demonstration - replace with df_final filtering
    time_data = pd.DataFrame({
        "Year": range(1990, 2024),
        "Housing Trend": np.cumsum(np.random.randn(34) + 0.5),
        "Industry Trend": np.cumsum(np.random.randn(34) + 0.4)
    }).set_index("Year")

    st.markdown(f"**Trends for {city_choice} ({ind_choice})**")
    st.line_chart(time_data[["Housing Trend", "Industry Trend"]], height=300)

    # 3. Fast-Growing vs Slow-Growing Comparison
    st.divider()
    st.subheader("🚀 Growth Cohort Comparison")
    cohort_view = st.selectbox("Compare Cohorts:", ["High Growth vs Low Growth", "High Diversity vs Low Diversity"])
    
    # Mock visualization for cohort comparison
    comparison_data = pd.DataFrame({
        "Year": range(2000, 2024),
        "High Growth Housing": np.linspace(100, 400, 24) + np.random.normal(0, 10, 24),
        "Low Growth Housing": np.linspace(100, 150, 24) + np.random.normal(0, 5, 24)
    }).set_index("Year")
    
    st.line_chart(comparison_data)

    # 4. PREMIUM FEATURE: Spot the Inflection
    st.divider()
    st.subheader("🎯 Automated Insights")
    
    if st.button("✨ Spot the Inflection Point"):
        st.balloons()
        inflection_year = 2012 # Example heuristic result
        st.success(f"**Inflection Detected:** In {city_choice}, {ind_choice} growth accelerated sharply in **{inflection_year}**.")
        
        st.markdown(f"""
        **What happened next?**
        - **Company Growth:** Jumped from 2% to 14% annually.
        - **Housing Lag:** Prices began a steep climb **18 months later**.
        - **Correlation:** $R^2 = 0.84$ following the inflection.
        """)
        
        # Highlight logic (visual representation)
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.lineplot(data=time_data, x=time_data.index, y="Housing Trend", ax=ax)
        plt.axvspan(inflection_year, inflection_year+3, color='yellow', alpha=0.3, label="Acceleration Window")
        plt.legend()
        st.pyplot(fig)
    
