import pytest
from mox.sampling import LHSStrategy
import jax.numpy as jnp

def test_lhs_strategy_valid_bounds():
    lower_bound = jnp.array([0, 0, 0])
    upper_bound = jnp.array([1, 1, 1])
    strategy = LHSStrategy(lower_bound, upper_bound)
    assert (strategy.lower_bound == lower_bound).all()
    assert (strategy.upper_bound == upper_bound).all()

def test_lhs_strategy_invalid_bounds():
    lower_bound = jnp.array([0, 0, 0])
    upper_bound = jnp.array([1, 1])  # Invalid: Mismatch in dimension
    with pytest.raises(ValueError) as error:
        LHSStrategy(lower_bound, upper_bound)
    assert str(error.value) == "Lower and upper bounds should have the same dimension."

def test_lhs_strategy_invalid_bounds_reverse_order():
    lower_bound = jnp.array([1, 1, 1])  # Invalid: Lower bound greater than upper bound
    upper_bound = jnp.array([0, 0, 0])
    with pytest.raises(ValueError) as error:
        LHSStrategy(lower_bound, upper_bound)
    assert str(error.value) == "The lower bound should be less than or equal to the upper bound."


