import jax.numpy as jnp
from mox.utils import tree_to_vector

def test_tree_to_vector_vectorises_single_sample_with_scalar_arrays():
    x_samples = [{
        'param1': jnp.array(1.0),
        'param2': jnp.array(3.0)
    }]
    assert jnp.array_equal(tree_to_vector(x_samples), jnp.array([1., 3.]))

def test_tree_to_vector_vectorises_single_sample_with_1d_arrays():
    x_samples = [{
        'param1': jnp.array([1.0, 2.0]),
        'param2': jnp.array([3.0, 4.0])
    }]
    assert jnp.array_equal(
        tree_to_vector(x_samples),
        jnp.array([1., 2., 3., 4.])
    )
