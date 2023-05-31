import pytest
import jax.numpy as jnp
from jax import random
from mox.sampling import sample, LHSStrategy, strategy_iterator
from scipy.stats.qmc import LatinHypercube
from unittest.mock import Mock, patch
from collections import OrderedDict

key = random.PRNGKey(42)

def test_sample_empty_strategy():
    with pytest.raises(ValueError) as error:
        sample([], 10, key)
    assert str(error.value) == "The strategy list should contain at least one strategy object."

def test_sample_non_iterable_strategy():
    with pytest.raises(TypeError) as error:
        sample(123, 10, key)
    assert str(error.value) == "The strategy parameter should be iterable, such as a list or tuple, containing strategy objects."

def test_sample_invalid_strategy_object_type():
    with pytest.raises(TypeError) as error:
        sample([LHSStrategy(1, 2), 123], 10, key)
    assert str(error.value) == "Invalid Strategy object."

def test_strategy_iterator_with_dictionary_iterates_over_all_strategies():
    strategy = OrderedDict([
        ("param1", LHSStrategy([0, 0], [1, 1])),
        ("param2", LHSStrategy([-1, -1], [1, 1]))
    ])
    assert list(strategy_iterator(strategy)) == list((strategy.values()))

def test_sample_calls_latin_hypercube_with_correct_dimensions():
    num_dimensions = 9
    n = 3
    strategy = [
        LHSStrategy(jnp.array([0, 0, 0]), jnp.array([1, 1, 1]))
        for _ in range(3)
    ]

    mock_latin_hypercube = Mock(wraps=LatinHypercube)
    with patch('mox.sampling.LatinHypercube', mock_latin_hypercube):
        sample(strategy, 10, key)

    mock_latin_hypercube.assert_called_once_with(d=num_dimensions, seed=42)

def test_sample_with_dictionary_strategy_calls_latin_hypercube_with_correct_dimensions():
    num_dimensions = 4
    strategy = {
        "param1": LHSStrategy(jnp.array([0, 0]), jnp.array([1, 1])),
        "param2": LHSStrategy(jnp.array([-1, -1]), jnp.array([1, 1]))
    }

    mock_latin_hypercube = Mock(wraps=LatinHypercube)
    with patch('mox.sampling.LatinHypercube', mock_latin_hypercube):
        sample(strategy, 10, key)

    mock_latin_hypercube.assert_called_once_with(d=num_dimensions, seed=42)

def test_sample_with_nested_list_strategy_calls_latin_hypercube_with_correct_dimensions():
    num_dimensions = 9
    strategy = [
        LHSStrategy(jnp.array([0, 0, 0]), jnp.array([1, 1, 1])),
        [
            LHSStrategy(jnp.array([-1, -1, -1]), jnp.array([0, 0, 0])),
            LHSStrategy(jnp.array([2, 2, 2]), jnp.array([3, 3, 3]))
        ]
    ]

    mock_latin_hypercube = Mock(wraps=LatinHypercube)
    with patch('mox.sampling.LatinHypercube', mock_latin_hypercube):
        sample(strategy, 10, key)

    mock_latin_hypercube.assert_called_once_with(d=num_dimensions, seed=42)

def test_sample_with_mixed_strategy_types_calls_latin_hypercube_with_correct_dimensions():
    num_dimensions = 7
    strategy = [
        LHSStrategy(jnp.array([0, 0]), jnp.array([1, 1])),
        {
            "param1": LHSStrategy(jnp.array([-1, -1, -1]), jnp.array([0, 0, 0])),
            "param2": LHSStrategy(jnp.array([2, 2]), jnp.array([3, 3]))
        }
    ]

    mock_latin_hypercube = Mock(wraps=LatinHypercube)
    with patch('mox.sampling.LatinHypercube', mock_latin_hypercube):
        sample(strategy, 10, key)

    mock_latin_hypercube.assert_called_once_with(d=num_dimensions, seed=42)
