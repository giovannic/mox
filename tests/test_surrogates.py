import jax.numpy as jnp
from jax.tree_util import tree_structure
from jax import jit
from mox.surrogates import summary, safe_summary, _recover
from flax.linen.module import _freeze_attr
from mox.utils import tree_to_vector
from utils import assert_tree_equal

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
        'param1': jnp.array([[2.0, 3.0]]),
        'param2': {
            'subparam1': jnp.array([[5.5], [7.5]]),
            'subparam2': jnp.array([[10.5]])
        }
    }

    expected_std = {
        'param1': jnp.array([[1.0, 1.0]]),
        'param2': {
            'subparam1': jnp.array([[0.5], [0.5]]),
            'subparam2': jnp.array([[1.118034]])
        }
    }

    assert_tree_equal(mean, expected_mean)
    assert_tree_equal(std, expected_std)

def test_summary_works_for_zero_variance():
    samples = {
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'param2': {
            'subparam1': jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            'subparam2': jnp.array([[0.0, 0.0], [0.0, 0.0]])
        }
    }

    axis = {
        'param1': 0,
        'param2': {
            'subparam1': 1,
            'subparam2': (0, 1)
        }
    }

    mean, std = safe_summary(summary(samples, axis))

    expected_mean = {
        'param1': jnp.array([[2.0, 3.0]]),
        'param2': {
            'subparam1': jnp.array([[5.5], [7.5]]),
            'subparam2': jnp.array([[0.]])
        }
    }

    expected_std = {
        'param1': jnp.array([[1.0, 1.0]]),
        'param2': {
            'subparam1': jnp.array([[0.5], [0.5]]),
            'subparam2': jnp.array([[1.]])
        }
    }

    assert_tree_equal(mean, expected_mean)
    assert_tree_equal(std, expected_std)



def test_vectorisation_works_for_dictionary_parameters():
    x_samples = [{
        'param1': jnp.array([[1.0, 2.0]]),
        'param2': jnp.array([[3.0, 4.0]])
    }]

    x_vec = tree_to_vector(x_samples)
    
    expected_x_vec = jnp.array([1., 2., 3., 4.])
    assert jnp.array_equal(x_vec, expected_x_vec)

def test_vectorisation_works_for_list_parameters():
    x_samples = [[
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0])
    ]]

    x_vec = tree_to_vector(x_samples)

    expected_x_vec = jnp.array([1., 2., 3., 4.])
    assert jnp.array_equal(x_vec, expected_x_vec)

def test_output_recovery_works_for_dictionary_output():
    y_vec = jnp.array([1.0, 2.0, 3.0])
    y_expected = {
        'output1': jnp.array([1.0]),
        'output2': jnp.array([2.0, 3.0])
    }
    y_shapes = [jnp.array([1]), jnp.array([2])]

    y = _recover(y_vec, (1, 3), y_shapes, tree_structure(y_expected))
    
    assert_tree_equal(y, y_expected)

def test_recovery_is_jitable():
    y_vec = jnp.array([1.0, 2.0, 3.0])
    y_expected = _freeze_attr({
        'output1': jnp.array([1.0]),
        'output2': jnp.array([2.0, 3.0])
    })
    y_shapes = [(1,), (2,)]

    y = jit(
        lambda y: _recover(y, (1, 3), y_shapes, tree_structure(y_expected))
    )(y_vec)
    
    assert_tree_equal(y, y_expected)

def test_output_recovery_works_for_list_output():
    y_vec = jnp.array([1.0, 2.0, 3.0])
    y_expected = _freeze_attr([
        jnp.array([1.0]),
        jnp.array([2.0, 3.0])
    ])
    y_shapes = [(1,), (2,)]

    y = _recover(y_vec, (1, 3), y_shapes, tree_structure(y_expected))

    assert_tree_equal(y, y_expected)
