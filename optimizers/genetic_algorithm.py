# optimizers/genetic_algorithm.py

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from .base_optimizer import BaseOptimizer


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Portfolio optimizer using Genetic Algorithm."""

    def __init__(self,
                 returns: pd.DataFrame,
                 risk_free_rate: float = 0.02,
                 population_size: int = 100,
                 mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8):
        """
        Initialize Genetic Algorithm optimizer.
        
        Args:
            returns (pd.DataFrame): Historical returns
            risk_free_rate (float): Risk-free rate
            population_size (int): Size of population in each generation
            mutation_rate (float): Probability of mutation for each gene
            crossover_rate (float): Probability of crossover between parents
        """
        super().__init__(returns, risk_free_rate)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        self.best_fitness = -np.inf
        self.history = []

    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize random population of portfolio weights.
        
        Returns:
            List[np.ndarray]: List of random portfolio weights
        """
        population = []
        for _ in range(self.population_size):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
            population.append(weights)
        return population

    def fitness_function(self,
                         weights: np.ndarray,
                         target_return: Optional[float] = None) -> float:
        """
        Calculate fitness of an individual.
        
        Args:
            weights (np.ndarray): Portfolio weights
            target_return (float, optional): Target return
            
        Returns:
            float: Fitness score
        """
        metrics = self.calculate_portfolio_metrics(weights)

        if target_return is not None:
            # Penalize deviation from target return
            return -abs(metrics['return'] - target_return)
        else:
            return metrics['sharpe_ratio']

    def select_parents(self,
                       population: List[np.ndarray],
                       fitness_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select two parents using tournament selection.
        
        Args:
            population (List[np.ndarray]): Current population
            fitness_scores (np.ndarray): Fitness scores for population
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected parents
        """
        def tournament_select():
            # Select random individuals for tournament
            candidates = np.random.choice(
                len(population), size=3, replace=False)
            tournament_fitness = fitness_scores[candidates]
            winner_idx = candidates[np.argmax(tournament_fitness)]
            return population[winner_idx]

        return tournament_select(), tournament_select()

    def crossover(self,
                  parent1: np.ndarray,
                  parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform crossover between parents.
        
        Args:
            parent1 (np.ndarray): First parent
            parent2 (np.ndarray): Second parent
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two offspring
        """
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Single-point crossover
        crossover_point = np.random.randint(1, self.n_assets)
        child1 = np.concatenate(
            [parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate(
            [parent2[:crossover_point], parent1[crossover_point:]])

        # Normalize weights
        child1 = child1 / np.sum(child1)
        child2 = child2 / np.sum(child2)

        return child1, child2

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Perform mutation on an individual.
        
        Args:
            individual (np.ndarray): Individual to mutate
            
        Returns:
            np.ndarray: Mutated individual
        """
        mutated = individual.copy()

        for i in range(self.n_assets):
            if np.random.random() < self.mutation_rate:
                # Add random noise to weight
                mutated[i] += np.random.normal(0, 0.1)

        # Ensure non-negative weights and normalize
        mutated = np.maximum(0, mutated)
        mutated = mutated / np.sum(mutated)

        return mutated

    def optimize(self,
                 n_generations: int = 100,
                 target_return: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Perform Genetic Algorithm optimization.
        
        Args:
            n_generations (int): Number of generations
            target_return (float, optional): Target return for the portfolio
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Optimal weights and corresponding metrics
        """
        # Initialize population
        population = self.initialize_population()
        self.history = []

        for generation in range(n_generations):
            # Calculate fitness for all individuals
            fitness_scores = np.array([
                self.fitness_function(ind, target_return) for ind in population
            ])

            # Update best solution
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = population[best_idx].copy()

            # Store history
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(fitness_scores)
            })

            # Create new population
            new_population = []

            while len(new_population) < self.population_size:
                # Select parents and create offspring
                parent1, parent2 = self.select_parents(
                    population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate offspring
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Update population
            population = new_population[:self.population_size]

        return self.best_individual, self.calculate_portfolio_metrics(self.best_individual)

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get history of optimization process.
        
        Returns:
            pd.DataFrame: Optimization history
        """
        return pd.DataFrame(self.history)
