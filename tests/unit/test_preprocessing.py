import numpy as np
import pytest


class TestSlidingWindow:
    """
    Validates temporal sequence slicing using a sliding window.
    """

    def test_basic_shape(self):
        """
        Verify that 2D input arrays partition into the expected 3D window tensors.
        """
        from utils.preprocessing import sliding_window
        data = np.random.randn(100, 5).astype(np.float32)
        w, _ = sliding_window(data, window_size=20, stride=1)
        assert w.shape == (81, 20, 5)

    def test_stride(self):
        """
        Verify that larger window stride steps yield fewer aggregated sequences.
        """
        from utils.preprocessing import sliding_window
        data = np.random.randn(100, 5).astype(np.float32)
        w1, _ = sliding_window(data, window_size=20, stride=1)
        w5, _ = sliding_window(data, window_size=20, stride=5)
        assert len(w5) < len(w1)

    def test_1d_input(self):
        """
        Verify that 1D input arrays are reshaped and windowed correctly.
        """
        from utils.preprocessing import sliding_window
        data = np.random.randn(50).astype(np.float32)
        w, _ = sliding_window(data, window_size=10, stride=1)
        assert w.shape[1] == 10
        assert w.shape[2] == 1

    def test_data_too_short_raises(self):
        """
        Verify that passing input arrays shorter than the window size raises a ValueError.
        """
        from utils.preprocessing import sliding_window
        data = np.random.randn(5, 3).astype(np.float32)
        with pytest.raises(ValueError):
            sliding_window(data, window_size=10)

    def test_include_targets(self):
        """
        Verify that target outputs are returned when include_targets is enabled.
        """
        from utils.preprocessing import sliding_window
        data = np.random.randn(50, 3).astype(np.float32)
        w, t = sliding_window(data, window_size=10, stride=1, include_targets=True)
        assert t is not None
        assert len(t) <= len(w)

    def test_window_content(self):
        """
        Verify window sequences values are structured in correct sequential order.
        """
        from utils.preprocessing import sliding_window
        data = np.arange(20).reshape(20, 1).astype(np.float32)
        w, _ = sliding_window(data, window_size=5, stride=1)
        np.testing.assert_array_equal(w[0, :, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(w[1, :, 0], [1, 2, 3, 4, 5])


class TestNormalization:
    """
    Validates Standard and MinMax data normalization scalers.
    """

    def test_standard_zero_mean(self):
        """
        Verify standard scaling centers feature columns to a zero mean.
        """
        from utils.preprocessing import normalize_data
        data = np.random.randn(200, 3).astype(np.float32) * 10 + 5
        norm, _ = normalize_data(data, method="standard")
        assert np.abs(norm.mean(axis=0)).max() < 0.1

    def test_standard_unit_variance(self):
        """
        Verify standard scaling normalizes feature columns to unit variance.
        """
        from utils.preprocessing import normalize_data
        data = np.random.randn(200, 3).astype(np.float32) * 10
        norm, _ = normalize_data(data, method="standard")
        assert np.abs(norm.std(axis=0) - 1.0).max() < 0.15

    def test_minmax_range(self):
        """
        Verify MinMax scaling bounds values strictly between 0 and 1.
        """
        from utils.preprocessing import normalize_data
        data = np.random.randn(200, 3).astype(np.float32)
        norm, _ = normalize_data(data, method="minmax")
        assert norm.min() >= -1e-7
        assert norm.max() <= 1.0 + 1e-7

    def test_fit_false_uses_existing_scaler(self):
        """
        Verify scaling validation sets using a pre-fit scaling parameter structure.
        """
        from utils.preprocessing import normalize_data
        data = np.random.randn(100, 3).astype(np.float32)
        _, scaler = normalize_data(data, method="standard")
        new_data = np.random.randn(50, 3).astype(np.float32)
        norm, _ = normalize_data(new_data, method="standard", scaler=scaler, fit=False)
        assert norm.shape == (50, 3)

    def test_1d_input(self):
        """
        Verify 1D input array scaling behaves identically by auto-expanding dimensions.
        """
        from utils.preprocessing import normalize_data
        data = np.random.randn(100).astype(np.float32)
        norm, _ = normalize_data(data, method="standard")
        assert norm.shape == (100, 1)

    def test_unknown_method_raises(self):
        """
        Verify that requesting an unsupported normalization method raises ValueError.
        """
        from utils.preprocessing import normalize_data
        with pytest.raises(ValueError):
            normalize_data(np.random.randn(10, 2), method="unknown")


class TestOutlierDetection:
    """
    Validates statistical outlier detection methods (IQR, Z-Score, MAD).
    """

    def test_iqr_detects_outliers(self):
        """
        Verify that IQR outlier detection correctly flags isolated spike values.
        """
        from utils.preprocessing import detect_outliers
        data = np.zeros((100, 1), dtype=np.float32)
        data[50] = 100.0
        outliers = detect_outliers(data, method="iqr")
        assert outliers[50] == True

    def test_zscore_detects_outliers(self):
        """
        Verify that Z-Score outlier detection correctly flags isolated spike values.
        """
        from utils.preprocessing import detect_outliers
        data = np.random.randn(100, 1).astype(np.float32)
        data[0] = 50.0
        outliers = detect_outliers(data, method="zscore")
        assert outliers[0] == True

    def test_mad_detects_outliers(self):
        """
        Verify that MAD outlier detection correctly flags isolated spike values.
        """
        from utils.preprocessing import detect_outliers
        data = np.random.randn(100, 1).astype(np.float32)
        data[0] = 50.0
        outliers = detect_outliers(data, method="mad")
        assert outliers[0] == True

    def test_no_outliers_in_normal_data(self):
        """
        Confirm that no outliers are flagged in normal, low-variance data.
        """
        from utils.preprocessing import detect_outliers
        np.random.seed(42)
        data = np.random.randn(100, 1).astype(np.float32) * 0.1
        outliers = detect_outliers(data, method="zscore", threshold=5.0)
        assert outliers.sum() == 0

    def test_unknown_method_raises(self):
        """
        Verify that requesting an unsupported outlier detection method raises ValueError.
        """
        from utils.preprocessing import detect_outliers
        with pytest.raises(ValueError):
            detect_outliers(np.random.randn(10, 1), method="unknown")


class TestFillMissingValues:
    """
    Validates missing values (NaN) imputation strategies.
    """

    def test_linear_interpolation(self):
        """
        Verify linear interpolation resolves missing values continuously.
        """
        from utils.preprocessing import fill_missing_values
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        filled = fill_missing_values(data, method="linear")
        assert not np.any(np.isnan(filled))
        assert abs(filled[1] - 2.0) < 0.1

    def test_mean_fill(self):
        """
        Verify mean imputation replaces missing elements with average non-NaN values.
        """
        from utils.preprocessing import fill_missing_values
        data = np.array([1.0, np.nan, 3.0])
        filled = fill_missing_values(data, method="mean")
        assert not np.any(np.isnan(filled))

    def test_no_nans_passthrough(self):
        """
        Confirm that arrays without missing values are returned unchanged.
        """
        from utils.preprocessing import fill_missing_values
        data = np.array([1.0, 2.0, 3.0])
        filled = fill_missing_values(data, method="linear")
        np.testing.assert_array_equal(data, filled)


class TestSmoothSeries:
    """
    Validates smoothing operations (Moving Average, Exponential Smoothing).
    """

    def test_moving_average(self):
        """
        Verify moving average filters high-frequency noise.
        """
        from utils.preprocessing import smooth_series
        data = np.random.randn(100).astype(np.float32)
        smoothed = smooth_series(data, window_size=5, method="moving_average")
        assert smoothed.shape == data.shape
        assert smoothed.std() <= data.std() + 0.1

    def test_exponential(self):
        """
        Verify exponential smoothing produces correct shapes.
        """
        from utils.preprocessing import smooth_series
        data = np.random.randn(100).astype(np.float32)
        smoothed = smooth_series(data, window_size=5, method="exponential")
        assert smoothed.shape == data.shape

    def test_unknown_method_raises(self):
        """
        Verify that requesting an unsupported smoothing method raises ValueError.
        """
        from utils.preprocessing import smooth_series
        with pytest.raises(ValueError):
            smooth_series(np.random.randn(10), method="unknown")


class TestCreateFeatures:
    """
    Validates dynamic feature engineering (differences, lag, moving stats).
    """

    def test_diff_features(self):
        """
        Verify that diff feature construction doubles feature size.
        """
        from utils.preprocessing import create_features
        data = np.random.randn(50, 3).astype(np.float32)
        features = create_features(data, feature_types=["diff"])
        assert features.shape == (50, 6)

    def test_rolling_mean(self):
        """
        Verify rolling mean feature constructs correct output dimension shapes.
        """
        from utils.preprocessing import create_features
        data = np.random.randn(50, 3).astype(np.float32)
        features = create_features(data, feature_types=["rolling_mean"])
        assert features.shape == (50, 6)

    def test_multiple_features(self):
        """
        Verify combining multiple features scales dimensions linearly.
        """
        from utils.preprocessing import create_features
        data = np.random.randn(50, 3).astype(np.float32)
        features = create_features(data, feature_types=["diff", "rolling_mean", "rolling_std", "lag"])
        assert features.shape[0] == 50
        assert features.shape[1] == 3 * 5
