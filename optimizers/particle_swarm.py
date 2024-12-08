# optimizers/particle_swarm.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from .base_optimizer import BaseOptimizer


class Particle:
    """Represents a single particle in the swarm."""

    def __init__(self, n_dimensions: int):
        """
        Initialize particle with random position and velocity.
        
        Args:
            n_dimensions (int): Number of dimensions (assets)
        """
        # Initialize random position (weights)
        self.position = np.random.random(n_dimensions)
        self.position = self.position / np.sum(self.position)

        # Initialize random velocity
        self.velocity = np.random.randn(n_dimensions) * 0.1

        # Initialize best position
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf


class ParticleSwarmOptimizer(BaseOptimizer):
    """Portfolio optimizer using Particle Swarm Optimization."""

    def __init__(self,
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.02,
                 n_particles: int = 50,
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5):
        """
        Initialize PSO optimizer.
        
        Args:
            returns (pd.DataFrame): Historical returns
            risk_free_rate (float): Risk-free rate
            n_particles (int): Number of particles in the swarm
            inertia_weight (float): Weight for particle's velocity
            cognitive_weight (float): Weight for particle's own best position
            social_weight (float): Weight for swarm's best position
        """
        super().__init__(returns, risk_free_rate)
        self.n_particles = n_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.history = []

    def initialize_swarm(self):
        """Initialize the swarm with random particles."""
        self.particles = [Particle(self.n_assets)
                          for _ in range(self.n_particles)]

        # Initialize global best
        for particle in self.particles:
            fitness = self.fitness_function(particle.position)
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

    def fitness_function(self,
                         weights: np.ndarray,
                         target_return: Optional[float] = None) -> float:
        """
        Calculate fitness of a particle position.
        
        Args:
            weights (np.ndarray): Portfolio weights
            target_return (float, optional): Target return
            
        Returns:
            float: Fitness score
        """
        metrics = self.calculate_portfolio_metrics(weights)

        if target_return is not None:
            return -abs(metrics['return'] - target_return)
        else:
            return metrics['sharpe_ratio']

    def update_particle(self,
                        particle: Particle,
                        target_return: Optional[float] = None):
        """
        Update particle's velocity and position.
        
        Args:
            particle (Particle): Particle to update
            target_return (float, optional): Target return
        """
        # Random coefficients
        r1, r2 = np.random.random(2)

        # Update velocity
        cognitive_component = self.cognitive_weight * r1 * \
            (particle.best_position - particle.position)
        social_component = self.social_weight * r2 * \
            (self.global_best_position - particle.position)

        particle.velocity = (self.inertia_weight * particle.velocity +
                             cognitive_component + social_component)

        # Update position
        particle.position = particle.position + particle.velocity

        # Ensure non-negative weights and normalize
        particle.position = np.maximum(0, particle.position)
        particle.position = particle.position / np.sum(particle.position)

        # Update particle's best position
        fitness = self.fitness_function(particle.position, target_return)
        if fitness > particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

    def optimize(self,
                 n_iterations: int = 100,
                 target_return: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform Particle Swarm Optimization.
        
        Args:
            n_iterations (int): Number of iterations
            target_return (float, optional): Target return for the portfolio
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Optimal weights and corresponding metrics
        """
        # Initialize swarm
        self.initialize_swarm()
        self.history = []

        for iteration in range(n_iterations):
            # Update all particles
            for particle in self.particles:
                self.update_particle(particle, target_return)

            # Store history
            avg_fitness = np.mean([p.best_fitness for p in self.particles])
            self.history.append({
                'iteration': iteration,
                'global_best_fitness': self.global_best_fitness,
                'average_fitness': avg_fitness
            })

        return self.global_best_position, self.calculate_portfolio_metrics(self.global_best_position)

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get history of optimization process.
        
        Returns:
            pd.DataFrame: Optimization history
        """
        return pd.DataFrame(self.history)
