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

    