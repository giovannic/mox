import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_leaves
from jax import random
from mox.surrogates import summary, Vectoriser, Recover 
from flax.linen.module import _freeze_attr

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

def test_vectorisation_works_for_dictionary_parameters():
    x_samples = _freeze_attr([{
        'param1': jnp.array([[1.0, 2.0], [5.0, 6.0]]),
        'param2': jnp.array([[3.0, 4.0], [7.0, 8.0]])
    }])

    x_mean = [{'param1': jnp.array([2.0]), 'param2': jnp.array([4.0])}]
    x_std = [{'param1': jnp.array([1.0]), 'param2': jnp.array([1.0])}]

    vec = Vectoriser(x_mean, x_std)

    key = random.PRNGKey(42)
    params = vec.init(key, x_samples)
    x_vec = vec.apply(params, x_samples)
    
    expected_x_vec = jnp.array([[-1., 0., -1., 0.], [ 3., 4., 3., 4.]])
    assert jnp.array_equal(x_vec, expected_x_vec)

def test_vectorisation_works_for_list_parameters():
    x_samples = _freeze_attr([[
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0])
    ]])

    x_mean = [[jnp.array([2.0]), jnp.array([4.0])]]
    x_std = [[jnp.array([1.0]), jnp.array([1.0])]]

    vec = Vectoriser(x_mean, x_std)

    key = random.PRNGKey(42)
    params = vec.init(key, x_samples)
    x_vec = vec.apply(params, x_samples)

    expected_x_vec = jnp.array([[-1., -1.], [-0., 0.]])
    assert jnp.array_equal(x_vec, expected_x_vec)

def test_output_recovery_works_for_dictionary_output():
    y_mean = {'output1': jnp.array([2.0]), 'output2': jnp.array([4.5, 5.5])}
    y_std = {'output1': jnp.array([0.5]), 'output2': jnp.array([1.0, 1.0])}
    y_min = {'output1': jnp.array([0.0]), 'output2': jnp.array([0.0, 0.0])}
    y_max = {'output1': jnp.array([100.0]), 'output2': jnp.array([100.0, 100.0])}

    y_vec = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_expected = _freeze_attr({
        'output1': jnp.array([[2.5], [4.0]]),
        'output2': jnp.array([[6.5, 8.5], [9.5, 11.5]])
    })
    y_shapes = [jnp.array(leaf.shape[1:]) for leaf in tree_leaves(y_expected)]
    rec = Recover(y_shapes, y_mean, y_std, y_min, y_max)

    key = random.PRNGKey(42)
    params = rec.init(key, y_vec)
    y = rec.apply(params, y_vec)
    
    assert_tree_equal(y, y_expected)

def test_output_recovery_works_for_list_output():
    y_mean = [jnp.array([2.0]), jnp.array([4.5, 5.5])]
    y_std = [jnp.array([0.5]), jnp.array([1.0, 1.0])]
    y_min = [jnp.array([0.0]), jnp.array([0.0, 0.0])]
    y_max = [jnp.array([100.0]), jnp.array([100.0, 100.0])]

    y_vec = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_expected = _freeze_attr([
        jnp.array([[2.5], [4.0]]),
        jnp.array([[6.5, 8.5], [9.5, 11.5]])
    ])
    y_shapes = [jnp.array(leaf.shape[1:]) for leaf in tree_leaves(y_expected)]
    rec = Recover(y_shapes, y_mean, y_std, y_min, y_max)

    key = random.PRNGKey(42)
    params = rec.init(key, y_vec)
    y = rec.apply(params, y_vec)

    assert_tree_equal(y, y_expected)

def test_output_recovery_limits_outputs():
    y_mean = {'output1': jnp.array([2.0]), 'output2': jnp.array([4.5, 5.5])}
    y_std = {'output1': jnp.array([0.5]), 'output2': jnp.array([1.0, 1.0])}
    y_min = {'output1': jnp.array([0.0]), 'output2': jnp.array([0.0, 0.0])}
    y_max = {'output1': jnp.array([1.0]), 'output2': jnp.array([100.0, 100.0])}

    y_vec = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_expected = _freeze_attr({
        'output1': jnp.array([[1.], [1.]]),
        'output2': jnp.array([[6.5, 8.5], [9.5, 11.5]])
    })
    y_shapes = [jnp.array(leaf.shape[1:]) for leaf in tree_leaves(y_expected)]
    rec = Recover(y_shapes, y_mean, y_std, y_min, y_max)

    key = random.PRNGKey(42)
    params = rec.init(key, y_vec)
    y = rec.apply(params, y_vec)
    
    assert_tree_equal(y, y_expected)
