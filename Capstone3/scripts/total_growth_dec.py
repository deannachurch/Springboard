import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

# Function to classify growth trends
def classify_category_trends(df, categories, min_publications=10, freq='QE'):
    results = []
    
    for category in categories:
        # Get data for this category
        category_df = df[df['category'] == category]
        
        # Skip if too few publications
        if len(category_df) < min_publications:
            continue
        
        # Create time series
        time_series = category_df.groupby(pd.Grouper(key='published_date', freq=freq)).size()
        
        # Calculate year-over-year growth rates
        if len(time_series) >= 8:  # Need at least 2 years of data
            # Convert index to numeric (years as float) for regression
            years_numeric = np.array([(idx.year + (idx.month/12)) for idx in time_series.index])
            counts = time_series.values
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = linregress(years_numeric, counts)
            
            # Calculate average and recent counts
            overall_avg = time_series.mean()
            recent_avg = time_series.iloc[-4:].mean()  # Last year (4 quarters)
            
            # Calculate growth metrics
            growth_rate = (recent_avg / overall_avg - 1) * 100 if overall_avg > 0 else 0
            
            # Calculate trend strength and significance
            trend_strength = r_value**2  # R-squared value
            is_significant = p_value < 0.05
            
            # Classify the trend
            if not is_significant:
                trend = "Stable (No Clear Trend)"
            elif slope > 0:
                if growth_rate > 20:
                    trend = "Strong Growth"
                else:
                    trend = "Moderate Growth"
            else:
                if growth_rate < -20:
                    trend = "Strong Decline"
                else:
                    trend = "Moderate Decline"
            
            results.append({
                'Category': category,
                'Publication Count': len(category_df),
                'Average Publications (Quarterly)': overall_avg,
                'Recent Average (Last Year)': recent_avg,
                'Growth Rate (%)': growth_rate,
                'Trend': trend,
                'Slope': slope,
                'R-squared': trend_strength,
                'p-value': p_value,
                'Is Significant': is_significant
            })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        return results_df.sort_values('Growth Rate (%)', ascending=False)
    else:
        return results_df

def plot_category_trend(df, category, freq='QE'):
    """
    Create a properly formatted trend visualization for a category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    category : str
        The category to plot
    freq : str, optional
        Time frequency, by default 'QE'
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Get data for this category
    category_df = df[df['category'] == category]
    
    # Create time series
    time_series = category_df.groupby(pd.Grouper(key='published_date', freq=freq)).size()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the actual data points
    ax.plot(time_series.index, time_series.values, 'o-', color='blue', label='Actual Publications')
    
    # Calculate trend line
    x = np.arange(len(time_series))
    y = time_series.values
    
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trend = intercept + slope * x
    
    # Plot trend line
    ax.plot(time_series.index, trend, 'r--', 
            label=f'Trend Line (R²: {r_value**2:.2f}, p: {p_value:.4f})')
    
    # Add moving average if enough data
    if len(time_series) >= 4:
        ma = time_series.rolling(window=4).mean()
        ax.plot(ma.index, ma.values, color='green', label='1-Year Moving Average')
    
    # Format the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Publication Count')
    ax.set_title(f"Publication Trend for {category}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis to show dates properly
    fig.autofmt_xdate()
    
    return fig