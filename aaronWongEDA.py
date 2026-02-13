import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
df = pd.read_csv("merged_companies_housing.csv")

print("="*80)
print("FIRMSCAPE: EXPLORATORY DATA ANALYSIS")
print("Industry Concentration & Housing Market Trends")
print("="*80)

# ============================================================================
# 1. DATA OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("1. DATA OVERVIEW")
print("="*80)

print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. CONVERT FIRM SIZE
# ============================================================================
print("\n" + "="*80)
print("2. FIRM SIZE CONVERSION")
print("="*80)

def convert_size(value):
    if isinstance(value, str):
        value = value.replace(',', '').strip()
        
        if '+' in value:
            value = value.replace('+', '')
            if 'K' in value:
                return float(value.replace('K', '')) * 1000
            else:
                return float(value)
        
        if '-' in value:
            low, high = value.split('-')
            
            def parse_number(x):
                x = x.strip()
                if 'K' in x:
                    return float(x.replace('K', '')) * 1000
                else:
                    return float(x)
            
            return (parse_number(low) + parse_number(high)) / 2
    
    return None

df['size_numeric'] = df['size'].apply(convert_size)

print(f"\nOriginal size values (sample):\n{df['size'].value_counts().head(10)}")
print(f"\nConverted size statistics:")
print(df['size_numeric'].describe())

# ============================================================================
# 3. GEOGRAPHIC DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("3. GEOGRAPHIC DISTRIBUTION")
print("="*80)

print(f"\nNumber of unique cities: {df['city_x'].nunique()}")
print(f"\nTop 20 cities by company count:")
city_counts = df['city_x'].value_counts().head(20)
print(city_counts)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Top cities by company count
city_counts.head(15).plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_xlabel('Number of Companies', fontsize=11)
ax1.set_ylabel('City', fontsize=11)
ax1.set_title('Top 15 Cities by Company Count', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# Companies per city distribution
ax2.hist(df['city_x'].value_counts(), bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Number of Companies', fontsize=11)
ax2.set_ylabel('Number of Cities', fontsize=11)
ax2.set_title('Distribution of Companies Across Cities', fontsize=13, fontweight='bold')
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

# ============================================================================
# 4. INDUSTRY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. INDUSTRY ANALYSIS")
print("="*80)

if 'industry' in df.columns:
    print(f"\nNumber of unique industries: {df['industry'].nunique()}")
    print(f"\nTop 15 industries by company count:")
    industry_counts = df['industry'].value_counts().head(15)
    print(industry_counts)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 7))
    industry_counts.plot(kind='barh', ax=ax, color='teal')
    ax.set_xlabel('Number of Companies', fontsize=11)
    ax.set_ylabel('Industry', fontsize=11)
    ax.set_title('Top 15 Industries by Company Count', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. FIRM SIZE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. FIRM SIZE ANALYSIS")
print("="*80)

print(f"\nFirm Size Distribution:")
print(df['size_numeric'].describe())

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax1.hist(df['size_numeric'].dropna(), bins=50, color='purple', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Firm Size', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Firm Sizes (Linear Scale)', fontsize=13, fontweight='bold')

# Log scale
ax2.hist(df['size_numeric'].dropna(), bins=50, color='darkgreen', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Firm Size', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of Firm Sizes (Log Scale)', fontsize=13, fontweight='bold')
ax2.set_xscale('log')

plt.tight_layout()
plt.show()

# ============================================================================
# 6. HOUSING PRICE INDEX ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. HOUSING PRICE INDEX ANALYSIS")
print("="*80)

print(f"\nHousing Price Index Statistics:")
print(df['hpi_index'].describe())

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
ax1.hist(df['hpi_index'].dropna(), bins=50, color='orangered', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Housing Price Index', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Housing Price Index', fontsize=13, fontweight='bold')

# Box plot
ax2.boxplot(df['hpi_index'].dropna(), vert=True)
ax2.set_ylabel('Housing Price Index', fontsize=11)
ax2.set_title('Housing Price Index Box Plot', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 7. CITY-LEVEL AGGREGATION
# ============================================================================
print("\n" + "="*80)
print("7. CITY-LEVEL AGGREGATION")
print("="*80)

city_stats = df.groupby('city_x').agg({
    'size_numeric': ['mean', 'median', 'std', 'count'],
    'hpi_index': ['mean', 'median', 'std']
}).reset_index()

city_stats.columns = ['city', 'avg_firm_size', 'median_firm_size', 'std_firm_size', 
                      'company_count', 'avg_hpi', 'median_hpi', 'std_hpi']
city_stats = city_stats.dropna()

print(f"\nCity Statistics Summary:")
print(city_stats.describe())

print(f"\nTop 10 cities by average firm size:")
print(city_stats.nlargest(10, 'avg_firm_size')[['city', 'avg_firm_size', 'company_count', 'avg_hpi']])

print(f"\nTop 10 cities by average HPI:")
print(city_stats.nlargest(10, 'avg_hpi')[['city', 'avg_hpi', 'company_count', 'avg_firm_size']])

print(f"\nTop 10 cities by company count:")
print(city_stats.nlargest(10, 'company_count')[['city', 'company_count', 'avg_firm_size', 'avg_hpi']])

# ============================================================================
# 8. CORRELATION ANALYSIS: FIRM SIZE vs HOUSING PRICES
# ============================================================================
print("\n" + "="*80)
print("8. FIRM SIZE vs HOUSING PRICE CORRELATION")
print("="*80)

# Calculate correlations
corr_pearson = city_stats['avg_firm_size'].corr(city_stats['avg_hpi'])
corr_spearman = city_stats['avg_firm_size'].corr(city_stats['avg_hpi'], method='spearman')
r, p_value = stats.pearsonr(city_stats['avg_firm_size'], city_stats['avg_hpi'])

print(f"\nPearson Correlation: {corr_pearson:.4f}")
print(f"Spearman Correlation: {corr_spearman:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Number of cities: {len(city_stats)}")

if p_value < 0.001:
    significance = "highly significant (p < 0.001)"
elif p_value < 0.01:
    significance = "very significant (p < 0.01)"
elif p_value < 0.05:
    significance = "significant (p < 0.05)"
else:
    significance = "not significant (p >= 0.05)"

print(f"Statistical significance: {significance}")

# Visualize
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(city_stats['avg_firm_size'],
                     city_stats['avg_hpi'],
                     s=city_stats['company_count']*2,
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=0.5,
                     c=city_stats['company_count'],
                     cmap='viridis')

# Add trend line
z = np.polyfit(np.log10(city_stats['avg_firm_size']), 
               city_stats['avg_hpi'], 1)
p = np.poly1d(z)
x_trend = np.logspace(np.log10(city_stats['avg_firm_size'].min()),
                      np.log10(city_stats['avg_firm_size'].max()), 100)
ax.plot(x_trend, p(np.log10(x_trend)), 
        "r--", alpha=0.8, linewidth=2, 
        label=f'Trend line (r={corr_pearson:.3f}, p={p_value:.4f})')

ax.set_xscale('log')
ax.set_xlabel('Average Firm Size (log scale)', fontsize=12)
ax.set_ylabel('Average Housing Price Index', fontsize=12)
ax.set_title('Firm Size vs Housing Price by City\n(bubble size = company count)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Company Count', fontsize=10)

plt.tight_layout()
plt.show()

# ============================================================================
# 9. INDUSTRY CONCENTRATION BY CITY
# ============================================================================
print("\n" + "="*80)
print("9. INDUSTRY CONCENTRATION BY CITY")
print("="*80)

if 'industry' in df.columns:
    # Calculate Herfindahl-Hirschman Index (HHI) for industry concentration
    city_industry = df.groupby(['city_x', 'industry']).size().reset_index(name='count')
    city_totals = df.groupby('city_x').size().reset_index(name='total')
    city_industry = city_industry.merge(city_totals, on='city_x')
    city_industry['share'] = city_industry['count'] / city_industry['total']
    city_industry['share_squared'] = city_industry['share'] ** 2
    
    hhi = city_industry.groupby('city_x')['share_squared'].sum().reset_index()
    hhi.columns = ['city', 'hhi']
    hhi = hhi.sort_values('hhi', ascending=False)
    
    print(f"\nTop 10 most concentrated cities (by HHI):")
    print(hhi.head(10))
    
    print(f"\nTop 10 most diversified cities (by HHI):")
    print(hhi.tail(10))
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.hist(hhi['hhi'], bins=50, color='salmon', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Herfindahl-Hirschman Index (HHI)', fontsize=11)
    ax.set_ylabel('Number of Cities', fontsize=11)
    ax.set_title('Distribution of Industry Concentration Across Cities\n(Higher HHI = More Concentrated)', 
                 fontsize=13, fontweight='bold')
    ax.axvline(hhi['hhi'].median(), color='red', linestyle='--', linewidth=2, label=f'Median = {hhi["hhi"].median():.3f}')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ============================================================================
# 10. KEY INSIGHTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("10. KEY INSIGHTS SUMMARY")
print("="*80)

print(f"""
Dataset Overview:
- Total companies: {len(df):,}
- Total cities: {df['city_x'].nunique()}
- Average companies per city: {len(df) / df['city_x'].nunique():.1f}

Firm Size:
- Mean: {df['size_numeric'].mean():.0f}
- Median: {df['size_numeric'].median():.0f}
- Range: {df['size_numeric'].min():.0f} - {df['size_numeric'].max():.0f}

Housing Prices:
- Mean HPI: {df['hpi_index'].mean():.1f}
- Median HPI: {df['hpi_index'].median():.1f}
- Range: {df['hpi_index'].min():.1f} - {df['hpi_index'].max():.1f}

Firm Size vs Housing Price Relationship:
- Correlation: {corr_pearson:.4f}
- Significance: {significance}
- Interpretation: {"Positive relationship - larger firms associated with higher housing prices" if corr_pearson > 0.3 else "Weak or negative relationship" if corr_pearson > 0 else "Negative relationship"}

Next Steps for Analysis:
1. Time-series analysis of housing price changes
2. Industry cluster identification (tech hubs, manufacturing centers, etc.)
3. Growth trajectory analysis (which cities are rising vs declining)
4. Predictive modeling for future growth cities
5. Comparison with historical case studies (Detroit, Bay Area, Austin, Seattle)
""")

print("="*80)
print("EDA COMPLETE")
print("="*80)