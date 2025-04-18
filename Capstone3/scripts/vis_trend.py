import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import Dict, Tuple, Any, List, Optional

from statsmodels.tsa.stattools import adfuller

def plot_total_publications(df: pd.DataFrame, freq: str = 'Q') -> plt.Figure:
    """
    Plot the total number of publications over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    freq : str, optional
        Time frequency ('Y' for yearly, 'Q' for quarterly, 'M' for monthly), by default 'Q'
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the plot
    """
    # Group by time period and count publications
    counts = df.groupby(pd.Grouper(key='published_date', freq=freq)).size()
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(ax=ax, marker='o')
    
    # Add a rolling average trend line (window size depends on frequency)
    window = 4 if freq == 'Y' else (4 if freq == 'Q' else 12)
    if len(counts) > window:  # Only if we have enough data points
        counts.rolling(window=window).mean().plot(ax=ax, color='red', 
                                                 linewidth=2, 
                                                 label=f'Trend (window={window})')
    
    # Add labels and title
    freq_label = 'Year' if freq == 'YE' else ('Quarter' if freq == 'QE' else 'Month')
    ax.set_xlabel(freq_label)
    ax.set_ylabel('Number of Publications')
    ax.set_title(f'Total Publications by {freq_label}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_top_categories(df: pd.DataFrame, top_n: int = 5, freq: str = 'Q') -> plt.Figure:
    """
    Plot publication counts for top categories over time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    top_n : int, optional
        Number of top categories to display, by default 5
    freq : str, optional
        Time frequency ('Y' for yearly, 'Q' for quarterly, 'M' for monthly), by default 'Q'
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the plot
    """
    # Find the top N categories
    top_categories = df['category'].value_counts().nlargest(top_n).index.tolist()
    
    # Filter the dataframe to include only top categories
    filtered_df = df[df['category'].isin(top_categories)]
    
    # Group by time period and category, then count
    counts = filtered_df.groupby([pd.Grouper(key='published_date', freq=freq), 'category']).size().unstack(fill_value=0)
    
    # Create time series plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each category
    for category in counts.columns:
        counts[category].plot(ax=ax, marker='o', label=category)
    
    # Add labels and title
    freq_label = 'Year' if freq == 'YE' else ('Quarter' if freq == 'QE' else 'Month')
    ax.set_xlabel(freq_label)
    ax.set_ylabel('Number of Publications')
    ax.set_title(f'Publication Counts for Top {top_n} Categories by {freq_label}')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Categories')
    
    return fig

def test_stationarity(time_series: pd.Series) -> Dict[str, Any]:
    """
    Test the stationarity of a time series using the Augmented Dickey-Fuller test.
    
    Parameters:
    -----------
    time_series : pd.Series
        The time series to test
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with test results
    """
    # Run ADF test
    result = adfuller(time_series.dropna())
    
    # Prepare results
    adf_test = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Is Stationary': result[1] <= 0.05
    }
    
    # Print results
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    print(f"Is Stationary: {result[1] <= 0.05}")
    
    return adf_test

def visualize_stationarity(time_series: pd.Series, window: int = 4) -> plt.Figure:
    """
    Create a visualization to help assess stationarity.
    
    Parameters:
    -----------
    time_series : pd.Series
        The time series to visualize
    window : int, optional
        Window size for rolling statistics, by default 4
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the plot
    """
    # Run stationarity test
    result = adfuller(time_series.dropna())
    is_stationary = result[1] <= 0.05
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original time series
    time_series.plot(ax=ax1, label='Original')
    ax1.set_title("Original Time Series")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rolling statistics
    time_series.plot(ax=ax2, label='Original', alpha=0.5)
    
    # Only calculate rolling statistics if we have enough data points
    if len(time_series) > window:
        rolling_mean = time_series.rolling(window=window).mean()
        rolling_std = time_series.rolling(window=window).std()
        
        rolling_mean.plot(ax=ax2, label=f'Rolling Mean (window={window})', color='red')
        rolling_std.plot(ax=ax2, label=f'Rolling Std (window={window})', color='green')
    
    ax2.set_title(f"Rolling Statistics - Stationarity: {'Yes' if is_stationary else 'No'}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add text box with test results
    result_text = (f"ADF Statistic: {result[0]:.4f}\n"
                  f"p-value: {result[1]:.4f}\n"
                  f"Stationary: {'Yes' if is_stationary else 'No'}")
    ax2.text(0.05, 0.95, result_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def analyze_top_categories(df: pd.DataFrame, top_n: int = 10, freq: str = 'Q') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze trends in top categories and test each for stationarity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    top_n : int, optional
        Number of top categories to analyze, by default 10
    freq : str, optional
        Time frequency ('Y', 'Q', 'M'), by default 'Q'
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - DataFrame with time series data for each top category
        - Dictionary with stationarity test results for each category
    """
    # Find the top N categories
    top_categories = df['category'].value_counts().nlargest(top_n).index.tolist()
    print(f"Analyzing top {top_n} categories:")
    for i, cat in enumerate(top_categories, 1):
        count = df[df['category'] == cat].shape[0]
        print(f"{i}. {cat} ({count} publications)")
    
    # Group by time period and category, then count
    counts_by_category = df.groupby([pd.Grouper(key='published_date', freq=freq), 'category']).size().unstack(fill_value=0)
    
    # Filter to include only top categories
    if counts_by_category.shape[1] > top_n:
        category_counts = counts_by_category[top_categories]
    else:
        category_counts = counts_by_category
    
    # Test stationarity for each category
    stationarity_results = {}
    
    for category in category_counts.columns:
        series = category_counts[category]
        
        # Skip if too few non-zero data points
        if (series > 0).sum() < 8:  # Need enough points for meaningful test
            stationarity_results[category] = {
                'error': 'Too few non-zero data points for reliable test',
                'Is Stationary': None
            }
            continue
        
        # Run ADF test
        try:
            result = adfuller(series.dropna())
            stationarity_results[category] = {
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Critical Values': result[4],
                'Is Stationary': result[1] <= 0.05
            }
        except Exception as e:
            stationarity_results[category] = {
                'error': str(e),
                'Is Stationary': None
            }
    
    return category_counts, stationarity_results

def plot_category_growth_rates(category_counts: pd.DataFrame, window: int = 4) -> plt.Figure:
    """
    Plot the growth rates of each category over time.
    
    Parameters:
    -----------
    category_counts : pd.DataFrame
        DataFrame with time series data for each category
    window : int, optional
        Window size for calculating percentage change, by default 4
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with the plot
    """
    # Calculate percentage change
    if window == 1:
        # Simple period-to-period change
        pct_change = category_counts.pct_change() * 100
    else:
        # Change compared to window periods ago
        pct_change = (category_counts / category_counts.shift(window) - 1) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot growth rate for each category
    for category in pct_change.columns:
        # Apply some smoothing for clearer visualization
        smoothed = pct_change[category].rolling(window=max(2, window//2)).mean()
        ax.plot(smoothed.index, smoothed, label=category)
    
    # Add a horizontal line at 0%
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Add labels and title
    time_period = 'year' if window == 4 else 'period'
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Growth Rate (% change over {window} {time_period}s)')
    ax.set_title(f'Growth Rates of Top Categories')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Categories')
    
    return fig

def plot_category_heatmap(category_counts: pd.DataFrame) -> plt.Figure:
    """
    Create a heatmap showing category trends over time.
    
    Parameters:
    -----------
    category_counts : pd.DataFrame
        DataFrame with time series data for each category
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with the heatmap
    """
    # Normalize data to show relative changes within each category
    normalized = category_counts.copy()
    for category in normalized.columns:
        max_val = normalized[category].max()
        if max_val > 0:  # Avoid division by zero
            normalized[category] = normalized[category] / max_val
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(normalized.T, cmap="YlGnBu", ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Category')
    ax.set_title('Category Trends Over Time (Normalized)')
    
    return fig

def plot_category_proportions(category_counts: pd.DataFrame) -> plt.Figure:
    """
    Plot how the proportion of each category changes over time.
    
    Parameters:
    -----------
    category_counts : pd.DataFrame
        DataFrame with time series data for each category
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure with the plot
    """
    # Calculate proportion of each category
    total_counts = category_counts.sum(axis=1)
    proportions = category_counts.div(total_counts, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot proportion for each category
    for category in proportions.columns:
        ax.plot(proportions.index, proportions[category], label=category)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion of Publications')
    ax.set_title('Changing Proportions of Categories Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(title='Categories')
    
    return fig

def identify_trending_categories(df: pd.DataFrame, 
                               num_periods: int = 4, 
                               min_threshold: int = 10,
                               freq: str = 'QE') -> pd.DataFrame:
    """
    Identify categories with the biggest changes in publication volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    num_periods : int, optional
        Number of most recent periods to consider, by default 4
    min_threshold : int, optional
        Minimum number of publications required in recent periods, by default 10
    freq : str, optional
        Time frequency ('Y', 'Q', 'M'), by default 'Q'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with categories and their growth rates, sorted by growth rate
    """
    # Group by time period and category
    counts = df.groupby([pd.Grouper(key='published_date', freq=freq), 'category']).size().unstack(fill_value=0)
    
    # Ensure we have enough data
    if len(counts) < num_periods * 2:
        print(f"Warning: Not enough time periods ({len(counts)}) for reliable trend analysis.")
        num_periods = max(2, len(counts) // 2)
    
    # Split data into recent and previous periods
    recent_data = counts.iloc[-num_periods:]
    previous_data = counts.iloc[-2*num_periods:-num_periods]
    
    # Calculate average counts for each period
    recent_avg = recent_data.mean()
    previous_avg = previous_data.mean()
    
    # Filter out categories with too few publications
    sufficient_volume = recent_avg >= min_threshold
    
    # Calculate growth rates
    growth_rates = ((recent_avg - previous_avg) / previous_avg.replace(0, 0.1)) * 100
    growth_rates = growth_rates[sufficient_volume]
    
    # Create a single DataFrame with all categories
    result = pd.DataFrame({
        'Category': growth_rates.index,
        'Growth Rate (%)': growth_rates.values,
        'Recent Avg': [recent_avg[cat] for cat in growth_rates.index],
        'Previous Avg': [previous_avg[cat] for cat in growth_rates.index]
    })
    
    # Sort by growth rate (descending)
    result = result.sort_values('Growth Rate (%)', ascending=False)
    
    # Add a trend indicator for clarity
    result['Trend'] = 'Growing'
    result.loc[result['Growth Rate (%)'] < 0, 'Trend'] = 'Declining'
    result.loc[result['Growth Rate (%)'].abs() < 5, 'Trend'] = 'Stable'  # Optional: mark nearly flat trends
    
    return result