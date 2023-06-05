import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_leaves
from mox.surrogates import summary 

def assert_tree_equal(x, y):
    assert tree_structure(x) == tree_structure(y)
    assert all(
        jnp.array_equal(lx, ly)
        for lx, ly in zip(tree_leaves(x), tree_leaves(y))
    )

def test_summary_gives_summary_for_mixed_samples_pytree():
    # Example test case with nested dictionary and jnp.array samples
    samples = {
        'param1': jnp.array([1.0, 2.0, 3.0]),
        'param2': {
            'subparam1': jnp.array([4.0, 5.0, 6.0]),
            'subparam2': jnp.array([7.0, 8.0, 9.0])
        }
    }

    mean, std = summary(samples)

    expected_mean = {
        'param1': jnp.array([2.0]),
        'param2': {
            'subparam1': jnp.array([5.0]),
            'subparam2': jnp.array([8.0])
        }
    }

    expected_std = {
        'param1': jnp.array([0.8164966]),
        'param2': {
            'subparam1': jnp.array([0.8164966]),
            'subparam2': jnp.array([0.8164966])
        }
    }

    assert mean == expected_mean
    assert std == expected_std

def test_summary_with_custom_axes():
    samples = {
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'param2': {
            'subparam1': jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            'subparam2': jnp.array([[9.0, 10.0], [11.0, 12.0]])
        }
    }

    axis = {
        'param1': 0,
        'param2': {
            'subparam1': 1,
            'subparam2': (0, 1)
        }
    }

    mean, std = summary(samples, axis)

    expected_mean = {
        'param1': jnp.array([2.0, 3.0]),
        'param2': {
            'subparam1': jnp.array([5.5, 7.5]),
            'subparam2': jnp.array(10.5)
        }
    }

    expected_std = {
        'param1': jnp.array([1.0, 1.0]),
        'param2': {
            'subparam1': jnp.array([0.5, 0.5]),
            'subparam2': jnp.array(1.118034)
        }
    }

    assert_tree_equal(mean, expected_mean)
    assert_tree_equal(std, expected_std)