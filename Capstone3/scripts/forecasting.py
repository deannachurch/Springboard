import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List, Optional, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import warnings
warnings.filterwarnings('ignore')

def train_test_split_time_series(series: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time series into training and testing sets.
    
    Parameters:
    -----------
    series : pd.Series
        The time series to split
    test_size : float, optional
        Proportion of data to use for testing, by default 0.2
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Training series and testing series
    """
    # Calculate the split point
    split_idx = int(len(series) * (1 - test_size))
    
    # Split the series
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    
    return train, test

def inverse_transform(transformed_series: pd.Series, 
                     original_series: pd.Series, 
                     transformation_method: str,
                     seasonal_period: int = 4) -> pd.Series:
    """
    Inverse transform a series back to its original scale.
    
    Parameters:
    -----------
    transformed_series : pd.Series
        The transformed time series
    original_series : pd.Series
        The original time series (needed for some transformations)
    transformation_method : str
        The transformation method used
    seasonal_period : int, optional
        The seasonal period for seasonal differencing, by default 4
        
    Returns:
    --------
    pd.Series
        The inverse-transformed series
    """
    if transformation_method == 'difference':
        # For difference, we need the first value of the original series
        # to reconstruct the levels
        last_original = original_series.iloc[0]
        reconstructed = pd.Series(index=transformed_series.index)
        reconstructed.iloc[0] = last_original
        
        for i in range(1, len(transformed_series)):
            reconstructed.iloc[i] = reconstructed.iloc[i-1] + transformed_series.iloc[i]
        
        return reconstructed
    
    elif transformation_method == 'log':
        # For log, we simply take the exponent
        # If we added a constant, we need to subtract it
        if (original_series <= 0).any():
            min_val = original_series[original_series > 0].min() if (original_series > 0).any() else 1
            constant = min_val / 10
            return np.exp(transformed_series) - constant
        else:
            return np.exp(transformed_series)
    
    elif transformation_method == 'log_difference':
        # First apply inverse of differencing, then inverse of log
        diff_reversed = pd.Series(index=transformed_series.index)
        
        # Get the log of the first value from original series
        if (original_series.iloc[0] <= 0).any():
            min_val = original_series[original_series > 0].min() if (original_series > 0).any() else 1
            constant = min_val / 10
            log_first = np.log(original_series.iloc[0] + constant)
        else:
            log_first = np.log(original_series.iloc[0])
            constant = 0
        
        diff_reversed.iloc[0] = log_first
        
        for i in range(1, len(transformed_series)):
            diff_reversed.iloc[i] = diff_reversed.iloc[i-1] + transformed_series.iloc[i]
        
        # Now apply inverse of log
        return np.exp(diff_reversed) - constant
    
    elif transformation_method == 'seasonal_difference':
        # For seasonal differencing, we need the first seasonal_period values
        reconstructed = pd.Series(index=transformed_series.index)
        
        # First seasonal_period values are from the original series
        for i in range(seasonal_period):
            if i < len(reconstructed):
                reconstructed.iloc[i] = original_series.iloc[i]
        
        # Remaining values are calculated using the seasonal differences
        for i in range(seasonal_period, len(transformed_series)):
            reconstructed.iloc[i] = reconstructed.iloc[i-seasonal_period] + transformed_series.iloc[i]
        
        return reconstructed
    
    elif transformation_method == 'twice_difference':
        # For second-order differencing, we need to invert twice
        # First, invert the second differencing
        first_diff = pd.Series(index=transformed_series.index)
        
        # We need the first two values of the first difference
        # Assuming original series was differenced to get first_diff_original
        first_diff_original = original_series.diff().dropna()
        
        # First value comes from the original first difference
        first_diff.iloc[0] = first_diff_original.iloc[0]
        
        # Calculate the rest of the first difference
        for i in range(1, len(transformed_series)):
            first_diff.iloc[i] = first_diff.iloc[i-1] + transformed_series.iloc[i]
        
        # Now invert the first differencing
        reconstructed = pd.Series(index=first_diff.index)
        reconstructed.iloc[0] = original_series.iloc[0]
        
        for i in range(1, len(first_diff)):
            reconstructed.iloc[i] = reconstructed.iloc[i-1] + first_diff.iloc[i]
        
        return reconstructed
    
    else:
        raise ValueError(f"Unknown transformation method: {transformation_method}")

def fit_arima_model(train_series: pd.Series, 
                   order: Tuple[int, int, int] = (1, 0, 0),
                   seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Any:
    """
    Fit an ARIMA or SARIMA model to the training data.
    
    Parameters:
    -----------
    train_series : pd.Series
        The training time series data
    order : Tuple[int, int, int], optional
        The (p, d, q) order of the ARIMA model, by default (1, 0, 0)
    seasonal_order : Optional[Tuple[int, int, int, int]], optional
        The (P, D, Q, S) seasonal order, by default None
        
    Returns:
    --------
    Any
        The fitted ARIMA or SARIMA model
    """
    if seasonal_order:
        # Fit SARIMA
        model = SARIMAX(train_series, 
                      order=order, 
                      seasonal_order=seasonal_order,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        fitted_model = model.fit(disp=False)
    else:
        # Fit ARIMA
        model = ARIMA(train_series, order=order)
        fitted_model = model.fit()
    
    return fitted_model

def fit_exponential_smoothing(train_series: pd.Series, 
                            seasonal_periods: int = 4,
                            trend: Optional[str] = 'add',
                            seasonal: Optional[str] = 'add') -> Any:
    """
    Fit an Exponential Smoothing model to the training data.
    
    Parameters:
    -----------
    train_series : pd.Series
        The training time series data
    seasonal_periods : int, optional
        The number of periods in a seasonal cycle, by default 4
    trend : Optional[str], optional
        The type of trend component, by default 'add'
    seasonal : Optional[str], optional
        The type of seasonal component, by default 'add'
        
    Returns:
    --------
    Any
        The fitted Exponential Smoothing model
    """
    model = ExponentialSmoothing(train_series, 
                               seasonal_periods=seasonal_periods,
                               trend=trend,
                               seasonal=seasonal,
                               use_boxcox=False)
    fitted_model = model.fit()
    return fitted_model

def evaluate_forecast(actual: pd.Series, 
                     forecast: pd.Series,
                     model_name: str) -> Dict[str, Any]:
    """
    Evaluate the accuracy of a forecast.
    
    Parameters:
    -----------
    actual : pd.Series
        The actual values
    forecast : pd.Series
        The forecasted values
    model_name : str
        Name of the model used
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with evaluation metrics
    """
    # Ensure the series have the same index
    common_idx = actual.index.intersection(forecast.index)
    actual = actual.loc[common_idx]
    forecast = forecast.loc[common_idx]
    
    # Calculate metrics
    mse = mean_squared_error(actual, forecast)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, forecast)
    
    # Try to calculate R² only if there's enough variability
    try:
        r2 = r2_score(actual, forecast)
    except:
        r2 = np.nan
    
    # Calculate mean absolute percentage error (MAPE)
    # Avoid division by zero
    mape = np.mean(np.abs((actual - forecast) / np.where(actual != 0, actual, 1))) * 100
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_forecast_results(original_train: pd.Series,
                        original_test: pd.Series,
                        forecast: pd.Series,
                        category: str,
                        model_name: str,
                        metrics: Dict[str, Any]) -> plt.Figure:
    """
    Plot the forecast results against actual data.
    
    Parameters:
    -----------
    original_train : pd.Series
        The original training data
    original_test : pd.Series
        The original test data
    forecast : pd.Series
        The forecast values
    category : str
        The category name
    model_name : str
        The name of the model used
    metrics : Dict[str, Any]
        Dictionary with evaluation metrics
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure with the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    original_train.plot(ax=ax, label='Training Data', color='blue')
    
    # Plot test data
    original_test.plot(ax=ax, label='Test Data (Actual)', color='green')
    
    # Plot forecast
    forecast.plot(ax=ax, label='Forecast', color='red', linestyle='--')
    
    # Add metrics as text
    metrics_text = (f"RMSE: {metrics['rmse']:.2f}\n"
                   f"MAE: {metrics['mae']:.2f}\n"
                   f"MAPE: {metrics['mape']:.2f}%")
    
    ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes,
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Publication Count')
    ax.set_title(f"Forecast for {category} using {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def compare_forecast_models(category: str,
                          transformed_series: pd.Series,
                          original_series: pd.Series,
                          transformation_method: str,
                          test_size: float = 0.2,
                          freq: str = 'Q') -> Dict[str, Any]:
    """
    Compare different forecasting models for a specific category.
    
    Parameters:
    -----------
    category : str
        The category name
    transformed_series : pd.Series
        The transformed (stationary) time series
    original_series : pd.Series
        The original (non-transformed) time series
    transformation_method : str
        The transformation method used
    test_size : float, optional
        Proportion of data to use for testing, by default 0.2
    freq : str, optional
        Time frequency, by default 'Q'
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with comprehensive comparison results
    """
    print(f"\nComparing forecast models for: {category}")
    print(f"Transformation method: {transformation_method}")
    
    # Split the transformed series into training and testing sets
    train_transformed, test_transformed = train_test_split_time_series(transformed_series, test_size)
    
    # Get corresponding original data
    train_original = original_series.loc[train_transformed.index]
    test_original = original_series.loc[test_transformed.index]
    
    print(f"Training set size: {len(train_transformed)}")
    print(f"Test set size: {len(test_transformed)}")
    
    results = {}
    forecasts = {}
    figures = {}
    
    # Model 1: ARIMA
    try:
        # For a stationary series, ARIMA(1,0,0) is a good starting point
        # Adjust the order based on the transformation
        if transformation_method in ['difference', 'log_difference', 'seasonal_difference', 'twice_difference']:
            # Already differenced, so no need for the I component
            arima_order = (1, 0, 0)
        else:
            # May need differencing if transformation didn't involve differencing
            arima_order = (1, 0, 0)
        
        # Try to fit ARIMA model
        arima_model = fit_arima_model(train_transformed, order=arima_order)
        
        # Forecast transformed values
        arima_forecast_transformed = arima_model.predict(
            start=test_transformed.index[0],
            end=test_transformed.index[-1]
        )
        
        # Convert back to original scale
        arima_forecast = pd.Series(
            inverse_transform(arima_forecast_transformed, original_series, transformation_method),
            index=test_transformed.index
        )
        
        # Evaluate forecast
        arima_metrics = evaluate_forecast(test_original, arima_forecast, "ARIMA")
        
        # Plot results
        arima_fig = plot_forecast_results(
            train_original,
            test_original,
            arima_forecast,
            category,
            "ARIMA",
            arima_metrics
        )
        
        # Store results
        results['ARIMA'] = arima_metrics
        forecasts['ARIMA'] = arima_forecast
        figures['ARIMA'] = arima_fig
        
        print(f"ARIMA - RMSE: {arima_metrics['rmse']:.2f}, MAPE: {arima_metrics['mape']:.2f}%")
    
    except Exception as e:
        print(f"ARIMA model failed: {str(e)}")
        results['ARIMA'] = {'error': str(e)}
    
    # Model 2: Seasonal ARIMA (if we have enough data)
    if len(train_transformed) >= 8:
        try:
            # For quarterly data, typical seasonal period is 4
            seasonal_period = 4 if freq == 'Q' else (12 if freq == 'M' else 1)
            
            sarima_order = (1, 0, 0)
            sarima_seasonal_order = (1, 0, 0, seasonal_period)
            
            sarima_model = fit_arima_model(
                train_transformed, 
                order=sarima_order,
                seasonal_order=sarima_seasonal_order
            )
            
            # Forecast transformed values
            sarima_forecast_transformed = sarima_model.predict(
                start=test_transformed.index[0],
                end=test_transformed.index[-1]
            )
            
            # Convert back to original scale
            sarima_forecast = pd.Series(
                inverse_transform(sarima_forecast_transformed, original_series, transformation_method),
                index=test_transformed.index
            )
            
            # Evaluate forecast
            sarima_metrics = evaluate_forecast(test_original, sarima_forecast, "SARIMA")
            
            # Plot results
            sarima_fig = plot_forecast_results(
                train_original,
                test_original,
                sarima_forecast,
                category,
                "SARIMA",
                sarima_metrics
            )
            
            # Store results
            results['SARIMA'] = sarima_metrics
            forecasts['SARIMA'] = sarima_forecast
            figures['SARIMA'] = sarima_fig
            
            print(f"SARIMA - RMSE: {sarima_metrics['rmse']:.2f}, MAPE: {sarima_metrics['mape']:.2f}%")
        
        except Exception as e:
            print(f"SARIMA model failed: {str(e)}")
            results['SARIMA'] = {'error': str(e)}
    
    # Model 3: Exponential Smoothing
    try:
        # Determine if we should include trend and seasonal components
        # based on the data characteristics
        seasonal_period = 4 if freq == 'Q' else (12 if freq == 'M' else 1)
        
        if len(train_transformed) < 2 * seasonal_period:
            # Not enough data for seasonal component
            trend = 'add'
            seasonal = None
        else:
            trend = 'add'
            seasonal = 'add'
        
        # Fit Exponential Smoothing model
        es_model = fit_exponential_smoothing(
            train_transformed,
            seasonal_periods=seasonal_period,
            trend=trend,
            seasonal=seasonal
        )
        
        # Forecast transformed values
        es_forecast_transformed = es_model.predict(
            start=test_transformed.index[0],
            end=test_transformed.index[-1]
        )
        
        # Convert back to original scale
        es_forecast = pd.Series(
            inverse_transform(es_forecast_transformed, original_series, transformation_method),
            index=test_transformed.index
        )
        
        # Evaluate forecast
        es_metrics = evaluate_forecast(test_original, es_forecast, "Exponential Smoothing")
        
        # Plot results
        es_fig = plot_forecast_results(
            train_original,
            test_original,
            es_forecast,
            category,
            "Exponential Smoothing",
            es_metrics
        )
        
        # Store results
        results['ExponentialSmoothing'] = es_metrics
        forecasts['ExponentialSmoothing'] = es_forecast
        figures['ExponentialSmoothing'] = es_fig
        
        print(f"Exponential Smoothing - RMSE: {es_metrics['rmse']:.2f}, MAPE: {es_metrics['mape']:.2f}%")
    
    except Exception as e:
        print(f"Exponential Smoothing model failed: {str(e)}")
        results['ExponentialSmoothing'] = {'error': str(e)}
    
    # Find the best model based on RMSE
    valid_models = {k: v for k, v in results.items() if 'rmse' in v}
    
    if valid_models:
        best_model = min(valid_models.items(), key=lambda x: x[1]['rmse'])[0]
        print(f"\nBest model for {category}: {best_model}")
        print(f"RMSE: {results[best_model]['rmse']:.2f}, MAPE: {results[best_model]['mape']:.2f}%")
    else:
        best_model = None
        print(f"\nNo valid models for {category}")
    
    return {
        'category': category,
        'results': results,
        'forecasts': forecasts,
        'figures': figures,
        'best_model': best_model,
        'train_data': train_original,
        'test_data': test_original
    }
    
def forecast_future_periods(category: str,
                           best_model_name: str,
                           transformation_method: str,
                           transformed_series: pd.Series,
                           original_series: pd.Series,
                           n_periods: int = 8,
                           freq: str = 'Q') -> Tuple[pd.Series, plt.Figure]:
    """
    Forecast future periods for a specific category using the best model.
    
    Parameters:
    -----------
    category : str
        The category to forecast
    best_model_name : str
        Name of the best model to use for forecasting
    transformation_method : str
        The transformation method used
    transformed_series : pd.Series
        The transformed time series
    original_series : pd.Series
        The original time series
    n_periods : int, optional
        Number of periods to forecast ahead, by default 8
    freq : str, optional
        Time frequency, by default 'Q'
        
    Returns:
    --------
    Tuple[pd.Series, plt.Figure]
        Forecasted series and visualization
    """
    print(f"\nGenerating future forecast for {category} using {best_model_name}...")
    
    # Create future date index
    last_date = original_series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                 periods=n_periods, 
                                 freq=freq)
    
    # Fit model on entire dataset
    if best_model_name == 'ARIMA':
        if transformation_method in ['difference', 'log_difference', 'seasonal_difference', 'twice_difference']:
            arima_order = (1, 0, 0)
        else:
            arima_order = (1, 0, 0)
        
        model = fit_arima_model(transformed_series, order=arima_order)
        
        # Generate future forecast (transformed)
        forecast_transformed = model.predict(
            start=len(transformed_series),
            end=len(transformed_series) + n_periods - 1
        )
        
    elif best_model_name == 'SARIMA':
        seasonal_period = 4 if freq == 'Q' else (12 if freq == 'M' else 1)
        sarima_order = (1, 0, 0)
        sarima_seasonal_order = (1, 0, 0, seasonal_period)
        
        model = fit_arima_model(
            transformed_series, 
            order=sarima_order,
            seasonal_order=sarima_seasonal_order
        )
        
        # Generate future forecast (transformed)
        forecast_transformed = model.predict(
            start=len(transformed_series),
            end=len(transformed_series) + n_periods - 1
        )
        
    elif best_model_name == 'ExponentialSmoothing':
        seasonal_period = 4 if freq == 'Q' else (12 if freq == 'M' else 1)
        
        if len(transformed_series) < 2 * seasonal_period:
            trend = 'add'
            seasonal = None
        else:
            trend = 'add'
            seasonal = 'add'
        
        model = fit_exponential_smoothing(
            transformed_series,
            seasonal_periods=seasonal_period,
            trend=trend,
            seasonal=seasonal
        )
        
        # Generate future forecast (transformed)
        forecast_transformed = model.predict(
            start=len(transformed_series),
            end=len(transformed_series) + n_periods - 1
        )
    else:
        raise ValueError(f"Unknown model type: {best_model_name}")
    
    # Convert forecast index to match the frequency
    forecast_transformed.index = forecast_index
    
    # Inverse transform the forecast
    forecast_original = inverse_transform(
        forecast_transformed, 
        original_series, 
        transformation_method
    )
    forecast_original = pd.Series(forecast_original, index=forecast_index)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    original_series.plot(ax=ax, label='Historical Data', color='blue')
    
    # Plot forecast
    forecast_original.plot(ax=ax, label='Future Forecast', color='red', linestyle='--')
    
    # Add confidence interval (simple approximation)
    if len(original_series) > 10:
        # Calculate historical volatility
        volatility = original_series.pct_change().std() * np.sqrt(n_periods)
        upper_bound = forecast_original * (1 + 1.96 * volatility)
        lower_bound = forecast_original * (1 - 1.96 * volatility)
        
        # Ensure lower bound is not negative
        lower_bound = lower_bound.clip(lower=0)
        
        ax.fill_between(forecast_index, lower_bound, upper_bound, 
                       color='red', alpha=0.2, label='95% Confidence Interval')
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Publication Count')
    ax.set_title(f"Future Forecast for {category} using {best_model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return forecast_original, fig

def create_growth_rate_summary(categories: list,
                             future_forecasts: Dict[str, pd.Series],
                             original_data: pd.DataFrame,
                             n_periods: int,
                             freq: str = 'Q') -> pd.DataFrame:
    """
    Create a summary of growth rates for each category.
    
    Parameters:
    -----------
    categories : list
        List of category names
    future_forecasts : Dict[str, pd.Series]
        Dictionary of future forecasts for each category
    original_data : pd.DataFrame
        Original dataframe with publication data
    n_periods : int
        Number of periods forecasted
    freq : str, optional
        Time frequency, by default 'Q'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with growth rate summary
    """
    # Calculate current values (most recent period)
    current_values = {}
    future_values = {}
    growth_rates = {}
    
    for category in categories:
        if category in future_forecasts:
            # Get most recent value from original data
            category_df = original_data[original_data['category'] == category]
            current_series = category_df.groupby(pd.Grouper(key='published_date', freq=freq)).size()
            current_values[category] = current_series.iloc[-1]
            
            # Get final forecasted value
            future_values[category] = future_forecasts[category].iloc[-1]
            
            # Calculate growth rate
            growth_rates[category] = (future_values[category] / current_values[category] - 1) * 100
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Category': list(growth_rates.keys()),
        'Current Value': [current_values[cat] for cat in growth_rates.keys()],
        f'Forecast in {n_periods} periods': [future_values[cat] for cat in growth_rates.keys()],
        'Growth Rate (%)': [growth_rates[cat] for cat in growth_rates.keys()]
    })
    
    # Sort by growth rate (descending)
    summary = summary.sort_values('Growth Rate (%)', ascending=False)
    
    return summary