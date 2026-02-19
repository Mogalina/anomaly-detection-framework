import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def sliding_window(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    include_targets: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input time series data of shape (n_samples, n_features)
        window_size: Size of the sliding window
        stride: Step size between windows
        include_targets: If True, return next value as target
    
    Returns:
        Tuple of (windows, targets) if include_targets else (windows, None)
        windows shape: (n_windows, window_size, n_features)
        targets shape: (n_windows, n_features) if include_targets
    """
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size})")
    
    n_windows = (len(data) - window_size) // stride + 1
    n_features = data.shape[1] if data.ndim > 1 else 1
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    windows = []
    targets = [] if include_targets else None
    
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
        
        if include_targets and i + window_size < len(data):
            targets.append(data[i + window_size])
    
    windows = np.array(windows)
    
    if include_targets and targets:
        targets = np.array(targets)
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.flatten()
        return windows, targets
    
    return windows, None


def normalize_data(
    data: np.ndarray,
    method: str = "standard",
    scaler: Optional[any] = None,
    fit: bool = True
) -> Tuple[np.ndarray, any]:
    """
    Normalize data using specified method.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        method: Normalization method ('standard' or 'minmax')
        scaler: Pre-fitted scaler, if None creates new one
        fit: Whether to fit the scaler
    
    Returns:
        Tuple of (normalized_data, scaler)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    if scaler is None:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    if fit:
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)
    
    return normalized, scaler


def detect_outliers(
    data: np.ndarray,
    method: str = "iqr",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.
    
    Args:
        data: Input data
        method: Detection method ('iqr', 'zscore', or 'mad')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    outliers = np.zeros(len(data), dtype=bool)
    
    for i in range(data.shape[1]):
        col = data[:, i]
        
        if method == "iqr":
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            col_outliers = (col < lower) | (col > upper)
        
        elif method == "zscore":
            mean = np.mean(col)
            std = np.std(col)
            if std > 0:
                z_scores = np.abs((col - mean) / std)
                col_outliers = z_scores > threshold
            else:
                col_outliers = np.zeros(len(col), dtype=bool)
        
        elif method == "mad":
            median = np.median(col)
            mad = np.median(np.abs(col - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (col - median) / mad
                col_outliers = np.abs(modified_z_scores) > threshold
            else:
                col_outliers = np.zeros(len(col), dtype=bool)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outliers |= col_outliers
    
    return outliers


def fill_missing_values(
    data: np.ndarray,
    method: str = "linear",
    limit: Optional[int] = None
) -> np.ndarray:
    """
    Fill missing values in time series data.
    
    Args:
        data: Input data with missing values (NaN)
        method: Filling method ('linear', 'forward', 'backward', 'mean')
        limit: Maximum number of consecutive NaNs to fill
    
    Returns:
        Data with missing values filled
    """
    if not np.any(np.isnan(data)):
        return data
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    filled = data.copy()
    
    for i in range(data.shape[1]):
        col = filled[:, i]
        mask = np.isnan(col)
        
        if not np.any(mask):
            continue
        
        if method == "linear":
            # Linear interpolation
            indices = np.arange(len(col))
            valid_indices = indices[~mask]
            valid_values = col[~mask]
            
            if len(valid_values) > 0:
                col[mask] = np.interp(indices[mask], valid_indices, valid_values)
        
        elif method == "forward":
            # Forward fill
            pd.Series(col).fillna(method='ffill', limit=limit, inplace=True)
        
        elif method == "backward":
            # Backward fill
            pd.Series(col).fillna(method='bfill', limit=limit, inplace=True)
        
        elif method == "mean":
            # Fill with mean
            mean_val = np.nanmean(col)
            col[mask] = mean_val
        
        else:
            raise ValueError(f"Unknown filling method: {method}")
        
        filled[:, i] = col
    
    if squeeze:
        filled = filled.flatten()
    
    return filled


def smooth_series(
    data: np.ndarray,
    window_size: int = 5,
    method: str = "moving_average"
) -> np.ndarray:
    """
    Smooth time series data.
    
    Args:
        data: Input time series
        window_size: Size of smoothing window
        method: Smoothing method ('moving_average' or 'exponential')
    
    Returns:
        Smoothed data
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    
    smoothed = np.zeros_like(data)
    
    for i in range(data.shape[1]):
        col = data[:, i]
        
        if method == "moving_average":
            # Simple moving average
            kernel = np.ones(window_size) / window_size
            smoothed[:, i] = np.convolve(col, kernel, mode='same')
        
        elif method == "exponential":
            # Exponential moving average
            alpha = 2 / (window_size + 1)
            ema = np.zeros_like(col)
            ema[0] = col[0]
            
            for j in range(1, len(col)):
                ema[j] = alpha * col[j] + (1 - alpha) * ema[j - 1]
            
            smoothed[:, i] = ema
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    if squeeze:
        smoothed = smoothed.flatten()
    
    return smoothed


def create_features(
    data: np.ndarray,
    feature_types: List[str] = None
) -> np.ndarray:
    """
    Create additional features from time series data.
    
    Args:
        data: Input time series of shape (n_samples, n_features)
        feature_types: List of feature types to create
                      ('diff', 'rolling_mean', 'rolling_std', 'lag')
    
    Returns:
        Extended feature array
    """
    if feature_types is None:
        feature_types = ['diff', 'rolling_mean']
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    features = [data]
    
    for feature_type in feature_types:
        if feature_type == 'diff':
            diff = np.diff(data, axis=0, prepend=data[0:1])
            features.append(diff)
        
        elif feature_type == 'rolling_mean':
            rolling_mean = np.zeros_like(data)
            for i in range(data.shape[1]):
                rolling_mean[:, i] = pd.Series(data[:, i]).rolling(
                    window=5, min_periods=1
                ).mean().values
            features.append(rolling_mean)
        
        elif feature_type == 'rolling_std':
            rolling_std = np.zeros_like(data)
            for i in range(data.shape[1]):
                rolling_std[:, i] = pd.Series(data[:, i]).rolling(
                    window=5, min_periods=1
                ).std().fillna(0).values
            features.append(rolling_std)
        
        elif feature_type == 'lag':
            lag = np.roll(data, shift=1, axis=0)
            lag[0] = data[0]
            features.append(lag)
    
    return np.concatenate(features, axis=1)
