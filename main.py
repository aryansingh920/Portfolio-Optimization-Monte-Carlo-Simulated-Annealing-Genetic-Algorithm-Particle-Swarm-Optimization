# main.py

import argparse
import logging
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from config.config import CONFIG
from data.data_loader import DataLoader
from optimizers.monte_carlo import MonteCarloOptimizer
from optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from optimizers.genetic_algorithm import GeneticAlgorithmOptimizer
from optimizers.particle_swarm import ParticleSwarmOptimizer
from utils.metrics import PortfolioMetrics
from utils.visualization import PortfolioVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio Optimization Tool")

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=CONFIG.default_tickers,
        help="List of stock tickers"
    )

    parser.add_argument(
        "--period",
        default=CONFIG.data.default_period,
        help="Data period (e.g., '5y', '1y')"
    )

    parser.add_argument(
        "--optimizer",
        choices=["monte_carlo", "simulated_annealing",
                 "genetic", "particle_swarm", "all"],
        default="all",
        help="Optimization algorithm to use"
    )

    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=CONFIG.risk_free_rate,
        help="Risk-free rate"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for saving results"
    )

    return parser.parse_args()


def run_optimization(optimizer_name: str, returns: pd.DataFrame, risk_free_rate: float):
    """Run specified optimization algorithm."""
    logger.info(f"Running {optimizer_name} optimization...")

    if optimizer_name == "monte_carlo":
        optimizer = MonteCarloOptimizer(returns, risk_free_rate)
        weights, metrics = optimizer.optimize(
            n_portfolios=CONFIG.optimizer.mc_portfolios)
    elif optimizer_name == "simulated_annealing":
        optimizer = SimulatedAnnealingOptimizer(returns, risk_free_rate)
        weights, metrics = optimizer.optimize(
            max_iterations=CONFIG.optimizer.sa_max_iterations)
    elif optimizer_name == "genetic":
        optimizer = GeneticAlgorithmOptimizer(
            returns,
            risk_free_rate,
            population_size=CONFIG.optimizer.ga_population_size,
            mutation_rate=CONFIG.optimizer.ga_mutation_rate,
            crossover_rate=CONFIG.optimizer.ga_crossover_rate
        )
        weights, metrics = optimizer.optimize(
            n_generations=CONFIG.optimizer.ga_generations)
    elif optimizer_name == "particle_swarm":
        optimizer = ParticleSwarmOptimizer(
            returns,
            risk_free_rate,
            n_particles=CONFIG.optimizer.pso_particles,
            inertia_weight=CONFIG.optimizer.pso_inertia,
            cognitive_weight=CONFIG.optimizer.pso_cognitive,
            social_weight=CONFIG.optimizer.pso_social
        )
        weights, metrics = optimizer.optimize(
            n_iterations=CONFIG.optimizer.pso_iterations)

    return optimizer, weights, metrics


def save_results(output_dir: str, optimizer_name: str, weights: pd.Series, metrics: dict):
    """Save optimization results."""
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{optimizer_name}_results_{timestamp}.json"

    results = {
        "weights": weights.to_dict(),
        "metrics": metrics
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_file}")


def main():
    """Main function."""
    args = parse_args()

    # Load and preprocess data
    logger.info("Loading data...")
    data_loader = DataLoader()
    price_data = data_loader.load_data(args.tickers, period=args.period)
    returns_data = data_loader.calculate_returns()

    # Get market data for comparison
    market_returns, risk_free_rate = data_loader.get_market_data(
        risk_free_rate=args.risk_free_rate
    )

    # Initialize visualizer
    visualizer = PortfolioVisualizer(style=CONFIG.plot_style)

    # Run optimizations
    optimizers = (
        ["all"]
        if args.optimizer == "all"
        else [args.optimizer]
    )

    results = {}
    for opt_name in optimizers:
        optimizer, weights, metrics = run_optimization(
            opt_name, returns_data, risk_free_rate
        )
        results[opt_name] = {
            "weights": pd.Series(weights, index=args.tickers),
            "metrics": metrics
        }

        # Save results
        save_results(args.output_dir, opt_name,
                     results[opt_name]["weights"], metrics)

        # Create visualizations
        if hasattr(optimizer, "get_optimization_history"):
            history = optimizer.get_optimization_history()
            fig = visualizer.plot_optimization_convergence(history, opt_name)
            fig.savefig(Path(args.output_dir) / f"{opt_name}_convergence.png")

        fig = visualizer.plot_portfolio_weights(
            weights, args.tickers, f"{opt_name} Portfolio Allocation"
        )
        fig.savefig(Path(args.output_dir) / f"{opt_name}_allocation.png")

    # Compare results if multiple optimizers were used
    if len(results) > 1:
        comparison = pd.DataFrame({
            name: res["metrics"]
            for name, res in results.items()
        }).T

        comparison.to_csv(Path(args.output_dir) /
                          "optimization_comparison.csv")
        logger.info(
            "Optimization comparison saved to optimization_comparison.csv")


if __name__ == "__main__":
    main()
