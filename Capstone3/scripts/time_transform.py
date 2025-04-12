import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from typing import Dict, Tuple, Any, List, Optional

from statsmodels.tsa.stattools import adfuller

def transform_to_stationary(series: pd.Series, 
                          method: str = 'difference',
                          seasonal_period: int = 4) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Transform a non-stationary time series to make it stationary.
    
    Parameters:
    -----------
    series : pd.Series
        The non-stationary time series to transform
    method : str, optional
        The transformation method, one of: 'difference', 'log', 'log_difference',
        'seasonal_difference', 'twice_difference', by default 'difference'
    seasonal_period : int, optional
        The seasonal period for seasonal differencing, by default 4 (quarterly)
        
    Returns:
    --------
    transformed_series : pd.Series
        The transformed, more stationary series
    results : Dict[str, Any]
        Dictionary with transformation and stationarity test results
    """
    results = {
        'original_series': series,
        'transformation_method': method,
        'original_stats': {}
    }
    
    # Test stationarity of original series
    try:
        adf_result = adfuller(series.dropna())
        results['original_stats'] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4],
            'Is Stationary': adf_result[1] <= 0.05
        }
    except Exception as e:
        results['original_stats'] = {'error': str(e)}
    
    # Apply transformation
    if method == 'difference':
        transformed = series.diff().dropna()
        desc = "First Difference"
    
    elif method == 'log':
        # Add a small constant if there are zeros or negative values
        if (series <= 0).any():
            min_val = series[series > 0].min() if (series > 0).any() else 1
            series_adj = series + min_val/10
            transformed = np.log(series_adj)
            desc = f"Log Transformation (with adjustment for non-positive values: +{min_val/10:.4f})"
        else:
            transformed = np.log(series)
            desc = "Log Transformation"
    
    elif method == 'log_difference':
        # Calculate log first, then difference
        if (series <= 0).any():
            min_val = series[series > 0].min() if (series > 0).any() else 1
            series_adj = series + min_val/10
            log_series = np.log(series_adj)
        else:
            log_series = np.log(series)
        transformed = log_series.diff().dropna()
        desc = "Log + First Difference"
    
    elif method == 'seasonal_difference':
        transformed = series.diff(seasonal_period).dropna()
        desc = f"Seasonal Difference (period={seasonal_period})"
    
    elif method == 'twice_difference':
        # Apply differencing twice (for series with strong trends)
        first_diff = series.diff().dropna()
        transformed = first_diff.diff().dropna()
        desc = "Second-order Difference"
    
    else:
        raise ValueError(f"Unknown transformation method: {method}. "
                         f"Valid methods are 'difference', 'log', 'log_difference', "
                         f"'seasonal_difference', and 'twice_difference'.")
    
    # Test stationarity of transformed series
    try:
        if len(transformed) >= 8:  # Need enough data points for a meaningful test
            adf_result = adfuller(transformed.dropna())
            results['transformed_stats'] = {
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Critical Values': adf_result[4],
                'Is Stationary': adf_result[1] <= 0.05
            }
        else:
            results['transformed_stats'] = {
                'error': 'Too few data points for reliable test'
            }
    except Exception as e:
        results['transformed_stats'] = {'error': str(e)}
    
    results['transformation_description'] = desc
    
    return transformed, results

def find_best_transformation(series: pd.Series, category_name: str) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Try different transformations and find the one that best makes the series stationary.
    
    Parameters:
    -----------
    series : pd.Series
        The non-stationary time series to transform
    category_name : str
        Name of the category (for display purposes)
        
    Returns:
    --------
    best_series : pd.Series
        The best transformed series
    results : Dict[str, Any]
        Dictionary with comprehensive results from all transformations
    """
    methods = ['difference', 'log', 'log_difference', 'seasonal_difference', 'twice_difference']
    all_results = {}
    best_method = None
    best_p_value = 1.0
    best_transformed = None
    
    print(f"\nFinding best transformation for: {category_name}")
    
    for method in methods:
        try:
            transformed, results = transform_to_stationary(series, method=method)
            all_results[method] = results
            
            # Check if transformation was successful in making series stationary
            if 'transformed_stats' in results and 'p-value' in results['transformed_stats']:
                p_value = results['transformed_stats']['p-value']
                is_stationary = results['transformed_stats'].get('Is Stationary', False)
                
                print(f"  {method}: p-value = {p_value:.6f}, Stationary: {is_stationary}")
                
                # Update best method if this one is better
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_method = method
                    best_transformed = transformed
            else:
                print(f"  {method}: Could not determine stationarity")
                
        except Exception as e:
            print(f"  {method}: Error - {str(e)}")
            all_results[method] = {'error': str(e)}
    
    # If none of the transformations worked, default to first difference
    if best_transformed is None:
        print("  No transformation was successful. Defaulting to first difference.")
        best_transformed, results = transform_to_stationary(series, method='difference')
        best_method = 'difference'
    else:
        print(f"  Best method: {best_method} (p-value: {best_p_value:.6f})")
    
    summary = {
        'best_method': best_method,
        'best_p_value': best_p_value,
        'is_stationary': best_p_value <= 0.05,
        'all_results': all_results
    }
    
    return best_transformed, summary

def visualize_transformation(original: pd.Series, 
                           transformed: pd.Series, 
                           category: str,
                           method: str,
                           is_stationary: bool) -> plt.Figure:
    """
    Create a visualization comparing original and transformed series.
    
    Parameters:
    -----------
    original : pd.Series
        The original time series
    transformed : pd.Series
        The transformed time series
    category : str
        The category name
    method : str
        The transformation method used
    is_stationary : bool
        Whether the transformed series is stationary
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original series
    original.plot(ax=ax1, marker='o')
    ax1.set_title(f"Original Series: {category}")
    ax1.set_ylabel("Publication Count")
    ax1.grid(True, alpha=0.3)
    
    # Plot transformed series
    transformed.plot(ax=ax2, marker='o', color='green')
    status = "Stationary" if is_stationary else "Non-stationary"
    ax2.set_title(f"Transformed Series ({method}): {status}")
    ax2.set_ylabel("Transformed Value")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def transform_multiple_categories(df: pd.DataFrame, 
                                categories: List[str], 
                                freq: str = 'Q') -> Dict[str, Any]:
    """
    Transform multiple categories and find best transformation for each.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing publication data
    categories : List[str]
        List of categories to transform
    freq : str, optional
        Time frequency ('Y', 'Q', 'M'), by default 'Q'
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with results for each category
    """
    results = {}
    transformed_series = {}
    
    # Get time series for each category
    for category in categories:
        # Create time series for this category
        category_df = df[df['category'] == category]
        time_series = category_df.groupby(pd.Grouper(key='published_date', freq=freq)).size()
        
        # Skip if not enough data points
        if len(time_series) < 8:
            print(f"Skipping {category}: Not enough data points ({len(time_series)})")
            continue
        
        # Find best transformation
        transformed, summary = find_best_transformation(time_series, category)
        
        # Store results
        results[category] = summary
        transformed_series[category] = transformed
        
        # Visualize transformation
        fig = visualize_transformation(
            time_series, 
            transformed,
            category,
            summary['best_method'],
            summary['is_stationary']
        )
        # plt.show()  # Uncomment to display figures immediately
        # fig.savefig(f"{category}_transformation.png")  # Uncomment to save figures
    
    return {
        'results': results,
        'transformed_series': transformed_series
    }