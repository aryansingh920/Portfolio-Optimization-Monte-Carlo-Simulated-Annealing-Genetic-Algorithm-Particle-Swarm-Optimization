# optimizers/base_optimizer.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class BaseOptimizer(ABC):
    """Abstract base class for portfolio optimization algorithms."""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the optimizer with historical returns data.
        
        Args:
            returns (pd.DataFrame): Historical returns for each asset
            risk_free_rate (float): Annual risk-free rate
        """
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.assets = returns.columns
        self.risk_free_rate = risk_free_rate
        
        # Calculate mean returns and covariance matrix
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate key portfolio metrics for given weights.
        
        Args:
            weights (np.ndarray): Array of portfolio weights
            
        Returns:
            Dict[str, float]: Dictionary containing portfolio metrics
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(weights.T @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'return': portfolio_return,
            'risk': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def validate_weights(self, weights: np.ndarray) -> bool:
        """
        Validate if weights satisfy basic constraints.
        
        Args:
            weights (np.ndarray): Array of portfolio weights
            
        Returns:
            bool: True if weights are valid, False otherwise
        """
        if weights is None:
            return False
            
        if len(weights) != self.n_assets:
            return False
            
        if not np.isclose(np.sum(weights), 1.0):
            return False
            
        if np.any(weights < 0):  # No short-selling constraint
            return False
            
        return True
    
    @abstractmethod
    def optimize(self, **kwargs) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize the portfolio weights.
        
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Optimal weights and corresponding metrics
        """
        pass
    
    def get_efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier points.
        
        Args:
            n_points (int): Number of points to generate
            
        Returns:
            pd.DataFrame: Efficient frontier points with risk and return
        """
        target_returns = np.linspace(
            self.mean_returns.min(),
            self.mean_returns.max(),
            n_points
        )
        
        efficient_portfolios = []
        for target_return in target_returns:
            weights, metrics = self.optimize(target_return=target_return)
            if weights is not None:
                efficient_portfolios.append({
                    'return': metrics['return'],
                    'risk': metrics['risk'],
                    'sharpe_ratio': metrics['sharpe_ratio']
                })
        
        return pd.DataFrame(efficient_portfolios)
