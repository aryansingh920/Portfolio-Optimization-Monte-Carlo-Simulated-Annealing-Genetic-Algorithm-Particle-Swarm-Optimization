# tests/test_data_loader.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data.data_loader import DataLoader
from config.config import CONFIG


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'AAPL': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
        'GOOGL': 150 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
        'MSFT': 200 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
    }, index=dates)

    return data


@pytest.fixture
def data_loader():
    """Create DataLoader instance."""
    return DataLoader()


def test_load_data(data_loader, sample_price_data):
    """Test data loading functionality."""
    with patch('yfinance.download', return_value=pd.DataFrame({'Adj Close': sample_price_data})):
        data = data_loader.load_data(['AAPL', 'GOOGL', 'MSFT'], period='1y')

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ['AAPL', 'GOOGL', 'MSFT']
        assert len(data) > 0
        assert not data.isnull().any().any()


def test_calculate_returns(data_loader, sample_price_data):
    """Test returns calculation."""
    data_loader.data = sample_price_data

    # Test simple returns
    returns = data_loader.calculate_returns(freq='D', log_returns=False)
    assert isinstance(returns, pd.DataFrame)
    assert len(returns) == len(sample_price_data) - \
        1  # One less due to differencing

    # Test log returns
    log_returns = data_loader.calculate_returns(freq='D', log_returns=True)
    assert isinstance(log_returns, pd.DataFrame)
    assert len(log_returns) == len(sample_price_data) - 1


def test_get_market_data(data_loader):
    """Test market data retrieval."""
    mock_market_data = pd.DataFrame({
        'Adj Close': [100, 101, 102]
    }, index=pd.date_range('2020-01-01', periods=3))

    with patch('yfinance.download', return_value=mock_market_data):
        market_returns, risk_free_rate = data_loader.get_market_data()

        assert isinstance(market_returns, pd.Series)
        assert isinstance(risk_free_rate, float)
        assert risk_free_rate >= 0


def test_preprocess_data(data_loader, sample_price_data):
    """Test data preprocessing."""
    # Create sample returns with outliers and missing values
    returns = pd.DataFrame({
        'AAPL': [0.01, 0.02, np.nan, 0.1, -0.5],
        'GOOGL': [0.02, np.nan, 0.03, 0.2, 0.4],
        'MSFT': [0.01, 0.02, 0.03, np.nan, 0.3]
    })

    data_loader.returns = returns
    processed_returns = data_loader.preprocess_data(winsorize=True)

    assert isinstance(processed_returns, pd.DataFrame)
    assert not processed_returns.isnull().any().any()
    assert processed_returns.max().max() < 0.5  # Check winsorization
    assert processed_returns.min().min() > -0.5  # Check winsorization


def test_error_handling(data_loader):
    """Test error handling in DataLoader."""
    # Test calculating returns without data
    with pytest.raises(ValueError):
        data_loader.calculate_returns()

    # Test preprocessing without returns
    with pytest.raises(ValueError):
        data_loader.preprocess_data()


def test_different_frequencies(data_loader, sample_price_data):
    """Test returns calculation with different frequencies."""
    data_loader.data = sample_price_data

    # Test daily returns
    daily_returns = data_loader.calculate_returns(freq='D')
    assert len(daily_returns) > 0

    # Test monthly returns
    monthly_returns = data_loader.calculate_returns(freq='M')
    assert len(monthly_returns) < len(daily_returns)

    # Test weekly returns
    weekly_returns = data_loader.calculate_returns(freq='W')
    assert len(weekly_returns) < len(daily_returns)
    assert len(weekly_returns) > len(monthly_returns)
