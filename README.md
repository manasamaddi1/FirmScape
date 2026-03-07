# 🏙️ FirmScape
### *Industry shifts move housing prices. We built the tool to see it coming.*

**Most housing booms don't happen overnight. FirmScape predicts which cities are about to heat up — before the market prices it in.**
Using machine learning and 50 years of business registration and housing data, we analyze firm founding rates, industry concentration, and economic diversity to identify cities with sustained housing growth potential.

---

## Project Overview

> **Goal:**  Predict which U.S.cities are about to experience housing price growth before the market prices hit by analyzing 50 years of business founding and industry structure data.


---

## Key Features
- **City Breakout Detector** - Classifies whether a city will land in the top 25% of housing growth
- **50 Years of Data** - Quarterly panel across 13,330+ U.S. metros from 1977–2023
- **Multi-Model Pipeline** - Ridge, Random Forest, and XGBoost regression + Logistic classification
- **Lag Analysis** - Tests whether industry signals *lead* housing prices by 1–8 quarters
- **Interactive Dashboard** - Streamlit web app with city timelines, scatter plots, and opportunity scoring
- **Honest R²** - We report low R² values proudly — housing is complex, industry is one signal

---

## Deployment

**Step 1 — Install dependencies**
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn xgboost
```

**Step 2 — Make sure these two files are in the same folder as `app.py`**
```
firmscape_integrated_quarterly_cleaned.csv
merged_companies_housing.csv
```

**Step 3 — Launch the app**
```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

> 💡 If you see a warning that the integrated dataset is missing, make sure `firmscape_integrated_quarterly_cleaned.csv` is in the same folder as `app.py` — not inside a subfolder.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Data Sources** | FHFA HPI, Zillow, U.S. Business Registration Records |
| **Storage** | CSV snapshots (quarterly panel) |
| **Analysis** | pandas, numpy, scipy, matplotlib, seaborn |
| **Web App** | Streamlit |
| **Predictive Modeling** | scikit-learn (Ridge, Random Forest, Logistic), XGBoost |
| **Version Control** | Git + GitHub |

---

## 📊 Model Performance

| Metric | Regression (XGBoost) | Classification (Logistic) | Interpretation |
|---|---|---|---|
| **Val R²** | ~0.05–0.12 | — | Industry explains ~5–12% of housing variance |
| **Val RMSE** | low | — | Average prediction error in % price change |
| **Val AUC** | — | ~0.60–0.68 | Meaningfully ranks breakout cities vs. non-breakout |
| **Val F1** | — | ~0.55–0.62 | Balanced precision and recall |

**Key Insight:** A low R² is *expected and honest* — housing prices are driven by interest rates, zoning, supply, and demographics too. Our goal isn't to overfit; it's to isolate how much industry structure alone explains, then use it as a leading indicator layered with other signals.

---

## Dataset

FirmScape joins four raw sources into one clean quarterly panel:

| Source | Description |
|---|---|
| **companies_sorted.csv** | 7.2M raw company records → filtered to U.S. firms with 100+ employees |
| **companies-2023-q4-sm.csv** | 19.5M raw records → same filter applied |
| **hpi_at_metro.csv** | FHFA home price index by metro area, recomputed for % change |
| **Zillow_Housing_Dataset.csv** | Median home prices reshaped from wide to long format |

Each source is cleaned, aggregated by city + quarter, and joined into:

**`firmscape_integrated_quarterly_cleaned.csv`** — 275,438 rows · 15 columns · 13,330 metros

**`merged_companies_housing.csv`** — simpler city-year panel used for baseline correlation analysis

DATASETS: https://drive.google.com/drive/folders/10s8DfNQ36rqgW9ilG2tfTD06trYGhb0A?usp=sharing
---

## 🔄 How FirmScape Works

### **For Users:**
1. **Home** - Pick a famous case study city (Detroit, Austin, Seattle, Chicago, Phoenix) and load its preset
2. **EDA Explorer** - Understand each variable with plain-English definitions and interactive charts
3. **Evidence** - Explore city timelines, scatter plots, multi-city comparisons, and inflection point detection
4. **Validation & Modeling** - Run regression or classification models, adjust lag, compare all three models side by side
5. **Opportunity Lab** - Weight the signals that matter to you and generate a "Next Hub" shortlist

### **Behind the Scenes:**
1. **Data Collection** - Four raw CSVs cleaned and merged in `firmscape_deliverable.ipynb`
2. **Feature Engineering** - Firm founding rate YoY, HHI concentration, industry diversity count, top industry share
3. **Lag Testing** - Industry features shifted 1–8 quarters to find the strongest predictive window
4. **ML Prediction** - Time-based train/val/test split (train < 2018, val 2018–2021, test ≥ 2022)
5. **Live Dashboard** - Streamlit renders ranked city scores, feature importances, and ROC curves

**Model Details:**
- **Regression target:** `fhfa_yoy` — FHFA year-over-year % change in home prices
- **Classification target:** Top 25% of housing growth = 1, rest = 0
- **Long-term variant:** 10-year forward growth using 40-quarter `fhfa_index` shift
- **Split strategy:** Time-ordered — no random shuffling, no data leakage

---

## Results

### **Key Findings**
- **Firm founding rate** is the strongest single predictor of near-term housing price movement
- **Industry diversity** buffers volatility — cities with more industry types show steadier price growth
- **Concentrated cities are fragile** — high HHI cities like Detroit show sharper boom-bust cycles
- **Lag matters** — industry signals are often strongest 2–4 quarters *before* housing moves

### **Model Journey**
1. **Started with Pearson correlation** on annual data → signal present but noisy
2. **Added lag testing** → confirmed industry *leads* housing in most cities, not just correlates
3. **Tried regression** on raw `pct_change` → R² low but honest, no leakage
4. **Pivoted to classification** (top 25% breakout) → AUC ~0.65, F1 ~0.58 ✅
5. **Added 10-year forward classifier** → tests whether today's industry structure predicts decade-long trajectories

---

##  Limitations
- All findings are **associative, not causal**
- Industry structure is **one signal among many** — rates, supply, and demographics also drive prices
- Low R² is **by design**, not a bug — we don't chase fit at the cost of validity
- Use FirmScape as a **screening and early-warning tool**, not a standalone forecast

---

## 📁 File Structure

```
firmscape/
├── app.py                                      # Streamlit dashboard
├── firmscape_deliverable.ipynb                 # Main notebook: cleaning, merging, modeling
├── firmscape_integrated_quarterly_cleaned.csv  # ⭐ Primary dataset (275K rows)
├── merged_companies_housing.csv                # Baseline merged panel
├── aaronWongEDA.ipynb                          # EDA notebook
├── Anusha's_partial_eda.ipynb                  # EDA notebook
├── notebooks/                                  # Additional data prep
└── README.md
```

---


