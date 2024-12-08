# utils/visualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


class PortfolioVisualizer:
    """Class for creating portfolio analysis visualizations."""

    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer with plot style.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)

    def plot_efficient_frontier(self,
                                portfolios: pd.DataFrame,
                                optimal_portfolio: Optional[Dict] = None,
                                figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot the efficient frontier from Monte Carlo simulation.
        
        Args:
            portfolios (pd.DataFrame): DataFrame with portfolio risks and returns
            optimal_portfolio (Dict, optional): Optimal portfolio to highlight
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot all portfolios
        scatter = ax.scatter(portfolios['risk'],
                             portfolios['return'],
                             c=portfolios['sharpe_ratio'],
                             cmap='viridis',
                             marker='o',
                             alpha=0.5)

        # Plot optimal portfolio if provided
        if optimal_portfolio:
            ax.scatter(optimal_portfolio['risk'],
                       optimal_portfolio['return'],
                       color='red',
                       marker='*',
                       s=200,
                       label='Optimal Portfolio')

        # Add colorbar
        plt.colorbar(scatter, label='Sharpe Ratio')

        # Labels and title
        ax.set_xlabel('Risk (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()

        return fig

    def plot_optimization_convergence(self,
                                      history: pd.DataFrame,
                                      optimizer_name: str,
                                      figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot optimization convergence history.
        
        Args:
            history (pd.DataFrame): Optimization history
            optimizer_name (str): Name of optimizer
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot best and average fitness
        ax.plot(history['best_fitness'],
                label='Best Fitness', color=self.colors[0])
        if 'avg_fitness' in history.columns:
            ax.plot(history['avg_fitness'], label='Average Fitness',
                    color=self.colors[1], alpha=0.7)

        # Labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness Value')
        ax.set_title(f'{optimizer_name} Convergence History')
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3)

        return fig

    def plot_portfolio_weights(self,
                               weights: np.ndarray,
                               assets: List[str],
                               title: str = "Portfolio Allocation",
                               figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot portfolio weights as a pie chart.
        
        Args:
            weights (np.ndarray): Portfolio weights
            assets (List[str]): Asset names
            title (str): Plot title
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Filter out very small allocations
        threshold = 0.01
        significant_weights = weights > threshold

        # Create pie chart
        wedges, texts, autotexts = ax.pie(weights[significant_weights],
                                          labels=np.array(assets)[
            significant_weights],
            autopct='%1.1f%%',
            colors=self.colors)

        # Style
        plt.setp(autotexts, size=8, weight="bold")
        plt.setp(texts, size=10)

        # Title
        ax.set_title(title)

        return fig

    def plot_cumulative_returns(self,
                                portfolio_returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot cumulative returns over time.
        
        Args:
            portfolio_returns (pd.Series): Portfolio returns
            benchmark_returns (pd.Series, optional): Benchmark returns
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate cumulative returns
        cum_returns = (1 + portfolio_returns).cumprod()
        ax.plot(cum_returns, label='Portfolio', color=self.colors[0])

        if benchmark_returns is not None:
            cum_benchmark = (1 + benchmark_returns).cumprod()
            ax.plot(cum_benchmark, label='Benchmark', color=self.colors[1])

        # Labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Portfolio Performance')
        ax.legend()

        # Grid
        ax.grid(True, alpha=0.3)

        return fig

    def plot_rolling_metrics(self,
                             returns: pd.Series,
                             window: int = 252,
                             figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot rolling metrics (volatility, returns, Sharpe ratio).
        
        Args:
            returns (pd.Series): Portfolio returns
            window (int): Rolling window size
            figsize (tuple): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.tight_layout(pad=3.0)

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        axes[0].plot(rolling_vol, color=self.colors[0])
        axes[0].set_title('Rolling Volatility')
        axes[0].grid(True, alpha=0.3)

        # Rolling returns
        rolling_ret = returns.rolling(window).mean() * 252
        axes[1].plot(rolling_ret, color=self.colors[1])
        axes[1].set_title('Rolling Annual Returns')
        axes[1].grid(True, alpha=0.3)

        # Rolling Sharpe ratio
        rolling_sharpe = rolling_ret / rolling_vol
        axes[2].plot(rolling_sharpe, color=self.colors[2])
        axes[2].set_title('Rolling Sharpe Ratio')
        axes[2].grid(True, alpha=0.3)

        return fig
