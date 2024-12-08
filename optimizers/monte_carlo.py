# optimizers/monte_carlo.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .base_optimizer import BaseOptimizer


class MonteCarloOptimizer(BaseOptimizer):
    """Portfolio optimizer using Monte Carlo simulation."""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """Initialize Monte Carlo optimizer."""
        super().__init__(returns, risk_free_rate)
        self.best_sharpe_ratio = -np.inf
        self.best_weights = None
        self.all_results = []

    def generate_weights(self) -> np.ndarray:
        """
        Generate random weights that sum to 1.
        
        Returns:
            np.ndarray: Random portfolio weights
        """
        weights = np.random.random(self.n_assets)
        return weights / np.sum(weights)

    def optimize(self,
                 n_portfolios: int = 10000,
                 target_return: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform Monte Carlo simulation to find optimal portfolio.
        
        Args:
            n_portfolios (int): Number of random portfolios to generate
            target_return (float, optional): Target return for the portfolio
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Optimal weights and corresponding metrics
        """
        self.best_sharpe_ratio = -np.inf
        self.best_weights = None
        self.all_results = []

        for _ in range(n_portfolios):
            weights = self.generate_weights()
            metrics = self.calculate_portfolio_metrics(weights)

            # Store results
            self.all_results.append({
                'weights': weights,
                **metrics
            })

            # Update best portfolio based on optimization criteria
            if target_return is not None:
                # Find portfolio closest to target return with minimum risk
                if abs(metrics['return'] - target_return) < 0.001:
                    if (self.best_weights is None or
                            metrics['risk'] < self.calculate_portfolio_metrics(self.best_weights)['risk']):
                        self.best_weights = weights
            else:
                # Find portfolio with highest Sharpe ratio
                if metrics['sharpe_ratio'] > self.best_sharpe_ratio:
                    self.best_sharpe_ratio = metrics['sharpe_ratio']
                    self.best_weights = weights

        if self.best_weights is None:
            return None, {}

        return self.best_weights, self.calculate_portfolio_metrics(self.best_weights)

    def get_simulation_results(self) -> pd.DataFrame:
        """
        Get results of all simulated portfolios.
        
        Returns:
            pd.DataFrame: Results of Monte Carlo simulation
        """
        results_df = pd.DataFrame(self.all_results)

        # Add individual asset weights as columns
        weights_df = pd.DataFrame([r['weights'] for r in self.all_results],
                                  columns=self.assets)

        return pd.concat([results_df.drop('weights', axis=1), weights_df], axis=1)
