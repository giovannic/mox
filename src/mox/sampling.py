"""This module handles sampling of parameter spaces.

It provides abstractions for setting sampling strategies for each parameter."""
from typing import Union, Dict, List, Any, Callable
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from scipy.stats.qmc import LatinHypercube
from jaxtyping import PyTree, Array
import jax.numpy as jnp
from jax import random

class Strategy(ABC): # pylint: disable=too-few-public-methods
    """Strategy. Abstract base class for strategy objects"""

@dataclass
class LHSStrategy(Strategy):
    """LHSStrategy. Strategy object for sampling a parameter from a Latin
    Hypercube"""

    lower_bound: Array
    upper_bound: Array

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = jnp.array(lower_bound)
        self.upper_bound = jnp.array(upper_bound)

        if self.lower_bound.shape != self.upper_bound.shape:
            raise ValueError(
                "Lower and upper bounds should have the same dimension."
            )

        if (self.lower_bound > self.upper_bound).any():
            raise ValueError(
                "The lower bound should be less than or equal " +
                "to the upper bound."
            )

    def transform_samples(self, samples: Array):
        """transform_samples.

        :param samples:
        :type samples: Array
        """
        return jnp.array(
            samples * self.upper_bound.flatten() - self.lower_bound.flatten()
                + self.lower_bound.flatten()
        ).reshape((-1,) + self.lower_bound.shape)

@dataclass
class DistStrategy(Strategy):
    distribution: Any

    def sample(self, key, sample_shape: int):
        return self.distribution.sample(key, (sample_shape,))


ParamStrategy = Union[
    Strategy,
    Iterable['ParamStrategy'],
    Dict[str, 'ParamStrategy']
]

def sample(strategy: list[ParamStrategy], num: int, key) -> list[PyTree]:
    """sample. Sample from a list of parameter sampling strategies

    :param strategy:
    :type strategy: list[ParamStrategy]
    :param n:
    :type n: int
    :param key:
    :rtype: list[PyTree]
    """

    if not isinstance(strategy, Iterable):
        raise TypeError(
            "The strategy parameter should be iterable, such as a " +
            "list or tuple, containing strategy objects."
        )

    if len(strategy) == 0:
        raise ValueError(
            "The strategy list should contain at least one " +
            "strategy object."
        )

    # Initialise sampler for LHS strategies
    lhs_dims = jnp.array([
        jnp.prod(jnp.array(s.lower_bound.shape)) for s in strategy_iterator(strategy)
        if isinstance(s, LHSStrategy)
    ])

    if len(lhs_dims) > 0:
        key_i, key = random.split(key)
        sampler = LatinHypercube(d=int(lhs_dims.sum()), seed=int(key_i[1]))
        lhs_samples = _lhs_sample_generator(lhs_dims, sampler.random(num))

    # Create strategy sampling function
    def sample_strategy(strat, key):
        if isinstance(strat, LHSStrategy):
            return strat.transform_samples(next(lhs_samples)), key

        key, key_i = random.split(key)
        return strat.sample(key_i, num), key_i

    # Do the sampling for each leaf

    return _strategy_transformer(strategy, sample_strategy, key)[0]

def strategy_iterator(strategy: List[ParamStrategy]):
    """strategy_iterator.

    :param strategy:
    :type strategy: List[ParamStrategy]
    """
    stack = [strategy]

    while stack:
        current = stack.pop()

        if isinstance(current, Strategy):
            yield current

        elif isinstance(current, (list, tuple)):
            stack.extend(reversed(current))

        elif isinstance(current, dict):
            stack.extend(reversed(current.values()))

        else:
            raise TypeError("Invalid Strategy object.")


def _strategy_transformer(strategy: Any, fun: Callable, key: Any):
    if isinstance(strategy, Strategy):
        return fun(strategy, key)

    if isinstance(strategy, (list, tuple)):
        result = []
        for sub_strategy in strategy:
            v, key = _strategy_transformer(sub_strategy, fun, key)
            result.append(v)
        return result, key

    if isinstance(strategy, dict):
        result = {}
        for k, sub_strategy in strategy.items():
            v, key = _strategy_transformer(sub_strategy, fun, key)
            result[k] = v
        return result, key

    raise TypeError("Invalid Strategy object.")

def _lhs_sample_generator(dims: Array, samples: Array):
    current_dim = 0
    for dim in dims:
        yield samples[:, current_dim:current_dim+dim]
        current_dim += dim
