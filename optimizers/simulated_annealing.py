# optimizers/simulated_annealing.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Callable
from .base_optimizer import BaseOptimizer


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """Portfolio optimizer using Simulated Annealing algorithm."""

    def __init__(self,
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.02,
                 temp_schedule: Optional[Callable[[int, int], float]] = None):
        """
        Initialize Simulated Annealing optimizer.
        
        Args:
            returns (pd.DataFrame): Historical returns
            risk_free_rate (float): Risk-free rate
            temp_schedule (Callable, optional): Temperature scheduling function
        """
        super().__init__(returns, risk_free_rate)
        self.temp_schedule = temp_schedule or self.default_temp_schedule
        self.best_state = None
        self.best_metrics = None
        self.history = []

    @staticmethod
    def default_temp_schedule(step: int, max_steps: int) -> float:
        """
        Default temperature scheduling function.
        
        Args:
            step (int): Current step
            max_steps (int): Maximum number of steps
            
        Returns:
            float: Current temperature
        """
        return max(0.1, min(1, 1 - step / max_steps))

    def generate_neighbor(self,
                          current_weights: np.ndarray,
                          temperature: float) -> np.ndarray:
        """
        Generate neighboring solution by perturbing current weights.
        
        Args:
            current_weights (np.ndarray): Current portfolio weights
            temperature (float): Current temperature
            
        Returns:
            np.ndarray: New portfolio weights
        """
        # Scale perturbation by temperature
        perturbation = np.random.normal(0, temperature, size=self.n_assets)
        new_weights = current_weights + perturbation

        # Ensure non-negative weights
        new_weights = np.maximum(0, new_weights)

        # Normalize to sum to 1
        return new_weights / np.sum(new_weights)

    def acceptance_probability(self,
                               old_cost: float,
                               new_cost: float,
                               temperature: float) -> float:
        """
        Calculate probability of accepting new solution.
        
        Args:
            old_cost (float): Cost of current solution
            new_cost (float): Cost of new solution
            temperature (float): Current temperature
            
        Returns:
            float: Acceptance probability
        """
        if new_cost > old_cost:
            return 1.0
        return np.exp((new_cost - old_cost) / temperature)

    def optimize(self,
                 max_iterations: int = 10000,
                 target_return: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform Simulated Annealing optimization.
        
        Args:
            max_iterations (int): Maximum number of iterations
            target_return (float, optional): Target return for the portfolio
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Optimal weights and corresponding metrics
        """
        # Initialize with random weights
        current_weights = np.random.random(self.n_assets)
        current_weights = current_weights / np.sum(current_weights)
        current_metrics = self.calculate_portfolio_metrics(current_weights)

        self.best_state = current_weights
        self.best_metrics = current_metrics
        self.history = []

        for step in range(max_iterations):
            temperature = self.temp_schedule(step, max_iterations)

            # Generate neighbor solution
            neighbor_weights = self.generate_neighbor(
                current_weights, temperature)
            neighbor_metrics = self.calculate_portfolio_metrics(
                neighbor_weights)

            # Calculate cost (negative Sharpe ratio or distance from target return)
            if target_return is not None:
                current_cost = -abs(current_metrics['return'] - target_return)
                neighbor_cost = - \
                    abs(neighbor_metrics['return'] - target_return)
            else:
                current_cost = current_metrics['sharpe_ratio']
                neighbor_cost = neighbor_metrics['sharpe_ratio']

            # Decide whether to accept neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > np.random.random():
                current_weights = neighbor_weights
                current_metrics = neighbor_metrics

                # Update best solution if necessary
                if neighbor_cost > self.best_metrics['sharpe_ratio']:
                    self.best_state = neighbor_weights
                    self.best_metrics = neighbor_metrics

            # Store history
            self.history.append({
                'step': step,
                'temperature': temperature,
                'cost': current_cost,
                'best_cost': self.best_metrics['sharpe_ratio']
            })

        return self.best_state, self.best_metrics

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get history of optimization process.
        
        Returns:
            pd.DataFrame: Optimization history
        """
        return pd.DataFrame(self.history)
