# data/data_loader.py

import pandas as pd
import yfinance as yf
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


class DataLoader:
    """Class for loading and preprocessing financial data."""

    def __init__(self):
        """Initialize DataLoader."""
        self.data = None
        self.returns = None

    def load_data(self,
                  tickers: List[str],
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  period: str = "5y") -> pd.DataFrame:
        """
        Load historical price data for given tickers.
        
        Args:
            tickers (List[str]): List of stock tickers
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            period (str, optional): Period to download if dates not specified
            
        Returns:
            pd.DataFrame: Historical adjusted close prices
        """
        if start_date is None or end_date is None:
            data = yf.download(tickers, period=period)
        else:
            data = yf.download(tickers, start=start_date, end=end_date)

        # Extract adjusted close prices
        if len(tickers) == 1:
            self.data = data['Adj Close'].to_frame()
        else:
            self.data = data['Adj Close']

        self.data.columns = tickers
        return self.data

    def calculate_returns(self,
                          freq: str = 'D',
                          log_returns: bool = False) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            freq (str): Frequency of returns ('D' for daily, 'M' for monthly)
            log_returns (bool): If True, calculate log returns
            
        Returns:
            pd.DataFrame: Returns data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if log_returns:
            self.returns = np.log(self.data / self.data.shift(1))
        else:
            self.returns = self.data.pct_change()

        if freq != 'D':
            self.returns = self.returns.resample(freq).sum()

        self.returns = self.returns.dropna()
        return self.returns

    def get_market_data(self,
                        market_symbol: str = "^GSPC",
                        risk_free_rate: Optional[float] = None) -> Tuple[pd.DataFrame, float]:
        """
        Get market index data and risk-free rate.
        
        Args:
            market_symbol (str): Market index symbol
            risk_free_rate (float, optional): Custom risk-free rate
            
        Returns:
            Tuple[pd.DataFrame, float]: Market returns and risk-free rate
        """
        if risk_free_rate is None:
            # Get risk-free rate from 13-week Treasury Bill
            try:
                treasury = yf.download("^IRX", period="1d")
                risk_free_rate = float(treasury['Adj Close'].iloc[-1]) / 100
            except:
                risk_free_rate = 0.02  # Default to 2% if unable to fetch

        market_data = yf.download(market_symbol,
                                  start=self.data.index[0],
                                  end=self.data.index[-1])
        market_returns = market_data['Adj Close'].pct_change().dropna()

        return market_returns, risk_free_rate

    def preprocess_data(self,
                        winsorize: bool = True,
                        winsorize_limits: Tuple[float, float] = (0.05, 0.95)) -> pd.DataFrame:
        """
        Preprocess returns data by handling outliers and missing values.
        
        Args:
            winsorize (bool): Whether to winsorize returns
            winsorize_limits (Tuple[float, float]): Percentile limits for winsorization
            
        Returns:
            pd.DataFrame: Processed returns
        """
        if self.returns is None:
            raise ValueError(
                "No returns data. Call calculate_returns() first.")

        processed_returns = self.returns.copy()

        # Handle missing values
        processed_returns = processed_returns.fillna(method='ffill')
        processed_returns = processed_returns.fillna(method='bfill')

        if winsorize:
            for column in processed_returns.columns:
                series = processed_returns[column]
                lower = series.quantile(winsorize_limits[0])
                upper = series.quantile(winsorize_limits[1])
                processed_returns[column] = series.clip(
                    lower=lower, upper=upper)

        return processed_returns
