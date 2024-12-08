# utils/metrics.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats


class PortfolioMetrics:
    """Class for calculating various portfolio performance metrics."""

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.02):
        """
        Initialize with portfolio returns and weights.
        
        Args:
            returns (pd.DataFrame): Historical returns for assets
            weights (np.ndarray): Portfolio weights
            risk_free_rate (float): Annual risk-free rate
        """
        self.returns = returns
        self.weights = weights
        self.risk_free_rate = risk_free_rate
        self.portfolio_returns = self.calculate_portfolio_returns()

    def calculate_portfolio_returns(self) -> pd.Series:
        """Calculate historical portfolio returns."""
        return self.returns.dot(self.weights)

    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic portfolio metrics.
        
        Returns:
            Dict[str, float]: Dictionary of basic metrics
        """
        annual_factor = 252  # Assuming daily returns

        portfolio_mean = self.portfolio_returns.mean() * annual_factor
        portfolio_std = self.portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (portfolio_mean - self.risk_free_rate) / portfolio_std

        return {
            'annual_return': portfolio_mean,
            'annual_volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }

    def calculate_drawdown_metrics(self) -> Dict[str, float]:
        """
        Calculate drawdown-related metrics.
        
        Returns:
            Dict[str, float]: Dictionary of drawdown metrics
        """
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1

        return {
            'max_drawdown': drawdowns.min(),
            'avg_drawdown': drawdowns.mean(),
            'drawdown_std': drawdowns.std()
        }

    def calculate_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate risk-related metrics.
        
        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        returns = self.portfolio_returns

        # Semi-deviation (downside risk)
        downside_returns = returns[returns < 0]
        semi_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)

        # Value at Risk (VaR)
        var_95 = stats.norm.ppf(0.05, returns.mean(), returns.std())

        # Conditional VaR (CVaR) / Expected Shortfall
        cvar_95 = returns[returns <= var_95].mean()

        # Sortino Ratio
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        sortino_ratio = excess_returns / semi_deviation if semi_deviation != 0 else 0

        return {
            'semi_deviation': semi_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': sortino_ratio
        }

    def calculate_beta_metrics(self, market_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate market-related metrics.
        
        Args:
            market_returns (pd.Series): Market benchmark returns
            
        Returns:
            Dict[str, float]: Dictionary of market-related metrics
        """
        # Calculate Beta
        covariance = np.cov(self.portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        beta = covariance / market_variance

        # Calculate Alpha (Jensen's Alpha)
        market_return = market_returns.mean() * 252
        portfolio_return = self.portfolio_returns.mean() * 252
        alpha = portfolio_return - \
            (self.risk_free_rate + beta * (market_return - self.risk_free_rate))

        # Information Ratio
        active_returns = self.portfolio_returns - market_returns
        information_ratio = (active_returns.mean() * 252) / \
            (active_returns.std() * np.sqrt(252))

        # Treynor Ratio
        treynor_ratio = (portfolio_return - self.risk_free_rate) / \
            beta if beta != 0 else 0

        return {
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio
        }

    def get_all_metrics(self, market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate all portfolio metrics.
        
        Args:
            market_returns (pd.Series, optional): Market benchmark returns
            
        Returns:
            Dict[str, float]: Dictionary of all metrics
        """
        metrics = {}
        metrics.update(self.calculate_basic_metrics())
        metrics.update(self.calculate_drawdown_metrics())
        metrics.update(self.calculate_risk_metrics())

        if market_returns is not None:
            metrics.update(self.calculate_beta_metrics(market_returns))

        return metrics
