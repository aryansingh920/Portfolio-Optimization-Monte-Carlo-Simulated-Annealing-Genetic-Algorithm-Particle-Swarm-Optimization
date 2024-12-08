# config/config.py

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OptimizerConfig:
    """Configuration for optimization algorithms."""

    # Monte Carlo settings
    mc_portfolios: int = 10000

    # Simulated Annealing settings
    sa_max_iterations: int = 1000
    sa_initial_temp: float = 1.0
    sa_cooling_rate: float = 0.95

    # Genetic Algorithm settings
    ga_population_size: int = 100
    ga_generations: int = 100
    ga_mutation_rate: float = 0.01
    ga_crossover_rate: float = 0.8

    # Particle Swarm settings
    pso_particles: int = 50
    pso_iterations: int = 100
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5


@dataclass
class DataConfig:
    """Configuration for data handling."""

    # Data source settings
    default_source: str = "yahoo"
    default_period: str = "5y"
    default_interval: str = "1d"

    # Preprocessing settings
    winsorize: bool = True
    winsorize_limits: tuple = (0.05, 0.95)
    fill_method: str = "ffill"

    # Return calculation settings
    return_type: str = "simple"  # or "log"
    frequency: str = "D"  # D: daily, M: monthly


@dataclass
class PortfolioConfig:
    """Configuration for portfolio constraints."""

    # Basic constraints
    min_weight: float = 0.0  # No short-selling
    max_weight: float = 1.0
    total_weight: float = 1.0

    # Risk constraints
    max_volatility: Optional[float] = None
    min_sharpe_ratio: Optional[float] = None

    # Diversity constraints
    max_sector_exposure: Optional[Dict[str, float]] = None
    min_assets: Optional[int] = None
    max_assets: Optional[int] = None


@dataclass
class ProjectConfig:
    """Main configuration class."""

    # Risk-free rate
    risk_free_rate: float = 0.02

    # Default tickers for testing
    default_tickers: List[str] = None

    # Optimization settings
    optimizer: OptimizerConfig = OptimizerConfig()

    # Data settings
    data: DataConfig = DataConfig()

    # Portfolio constraints
    portfolio: PortfolioConfig = PortfolioConfig()

    # Visualization settings
    plot_style: str = "seaborn"
    figure_size: tuple = (12, 8)

    def __post_init__(self):
        if self.default_tickers is None:
            self.default_tickers = [
                "AAPL", "GOOGL", "MSFT", "AMZN", "META",
                "BRK-B", "JPM", "JNJ", "V", "PG"
            ]


# Create default configuration
CONFIG = ProjectConfig()
