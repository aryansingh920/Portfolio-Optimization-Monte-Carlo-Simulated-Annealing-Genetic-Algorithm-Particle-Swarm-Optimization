# tests/test_optimizers.py

import pytest
import pandas as pd
import numpy as np
from typing import Type

from optimizers.base_optimizer import BaseOptimizer
from optimizers.monte_carlo import MonteCarloOptimizer
from optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from optimizers.genetic_algorithm import GeneticAlgorithmOptimizer
from optimizers.particle_swarm import ParticleSwarmOptimizer


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.001, 0.02, len(dates)),
        'MSFT': np.random.normal(0.001, 0.02, len(dates))
    }, index=dates)
    return returns


@pytest.fixture(params=[
    MonteCarloOptimizer,
    SimulatedAnnealingOptimizer,
    GeneticAlgorithmOptimizer,
    ParticleSwarmOptimizer
])
def optimizer_class(request):
    """Parametrize test with different optimizer classes."""
    return request.param


def test_optimizer_initialization(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test optimizer initialization."""
    optimizer = optimizer_class(sample_returns, risk_free_rate=0.02)

    assert optimizer.returns.equals(sample_returns)
    assert optimizer.n_assets == len(sample_returns.columns)
    assert optimizer.risk_free_rate == 0.02
    assert isinstance(optimizer.mean_returns, pd.Series)
    assert isinstance(optimizer.cov_matrix, pd.DataFrame)


def test_portfolio_metrics(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test portfolio metrics calculation."""
    optimizer = optimizer_class(sample_returns)

    # Test with equal weights
    weights = np.array([1/3, 1/3, 1/3])
    metrics = optimizer.calculate_portfolio_metrics(weights)

    assert isinstance(metrics, dict)
    assert 'return' in metrics
    assert 'risk' in metrics
    assert 'sharpe_ratio' in metrics
    assert all(isinstance(v, float) for v in metrics.values())


def test_weight_validation(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test portfolio weight validation."""
    optimizer = optimizer_class(sample_returns)

    # Valid weights
    valid_weights = np.array([0.3, 0.3, 0.4])
    assert optimizer.validate_weights(valid_weights)

    # Invalid weights (sum != 1)
    invalid_weights1 = np.array([0.3, 0.3, 0.3])
    assert not optimizer.validate_weights(invalid_weights1)

    # Invalid weights (negative)
    invalid_weights2 = np.array([0.5, 0.8, -0.3])
    assert not optimizer.validate_weights(invalid_weights2)

    # Invalid weights (wrong length)
    invalid_weights3 = np.array([0.5, 0.5])
    assert not optimizer.validate_weights(invalid_weights3)


def test_optimization_results(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test optimization results."""
    optimizer = optimizer_class(sample_returns)

    if optimizer_class == MonteCarloOptimizer:
        weights, metrics = optimizer.optimize(n_portfolios=1000)
    elif optimizer_class == SimulatedAnnealingOptimizer:
        weights, metrics = optimizer.optimize(max_iterations=1000)
    elif optimizer_class == GeneticAlgorithmOptimizer:
        weights, metrics = optimizer.optimize(n_generations=50)
    else:  # ParticleSwarmOptimizer
        weights, metrics = optimizer.optimize(n_iterations=100)

    assert isinstance(weights, np.ndarray)
    assert len(weights) == optimizer.n_assets
    assert np.isclose(np.sum(weights), 1.0)
    assert all(w >= 0 for w in weights)
    assert isinstance(metrics, dict)


def test_efficient_frontier(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test efficient frontier generation."""
    optimizer = optimizer_class(sample_returns)

    frontier = optimizer.get_efficient_frontier(n_points=10)

    assert isinstance(frontier, pd.DataFrame)
    assert len(frontier) > 0
    assert all(col in frontier.columns for col in [
               'return', 'risk', 'sharpe_ratio'])


def test_optimization_with_target_return(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test optimization with target return constraint."""
    optimizer = optimizer_class(sample_returns)
    target_return = 0.001  # 0.1% daily return

    if optimizer_class == MonteCarloOptimizer:
        weights, metrics = optimizer.optimize(
            n_portfolios=1000, target_return=target_return)
    elif optimizer_class == SimulatedAnnealingOptimizer:
        weights, metrics = optimizer.optimize(
            max_iterations=1000, target_return=target_return)
    elif optimizer_class == GeneticAlgorithmOptimizer:
        weights, metrics = optimizer.optimize(
            n_generations=50, target_return=target_return)
    else:  # ParticleSwarmOptimizer
        weights, metrics = optimizer.optimize(
            n_iterations=100, target_return=target_return)

    assert isinstance(weights, np.ndarray)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.isclose(metrics['return'], target_return, rtol=0.1)


@pytest.mark.parametrize("optimizer_class", [
    MonteCarloOptimizer,
    SimulatedAnnealingOptimizer,
    GeneticAlgorithmOptimizer,
    ParticleSwarmOptimizer
])
def test_optimizer_convergence(optimizer_class: Type[BaseOptimizer], sample_returns):
    """Test optimizer convergence behavior."""
    optimizer = optimizer_class(sample_returns)

    # Store initial and final metrics
    if optimizer_class == MonteCarloOptimizer:
        weights, metrics = optimizer.optimize(n_portfolios=1000)
        if hasattr(optimizer, 'all_results'):
            assert len(optimizer.all_results) == 1000
    else:
        if optimizer_class == SimulatedAnnealingOptimizer:
            weights, metrics = optimizer.optimize(max_iterations=1000)
        elif optimizer_class == GeneticAlgorithmOptimizer:
            weights, metrics = optimizer.optimize(n_generations=50)
        else:  # ParticleSwarmOptimizer
            weights, metrics = optimizer.optimize(n_iterations=100)

        if hasattr(optimizer, 'history'):
            assert len(optimizer.history) > 0

            # Check if optimization improved the solution
            initial_fitness = optimizer.history[0].get('best_fitness')
            final_fitness = optimizer.history[-1].get('best_fitness')
            assert final_fitness >= initial_fitness
