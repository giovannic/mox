import jax.numpy as jnp
from mox.utils import tree_to_vector

def test_tree_to_vector_works_for_scalar_arrays():
    x_samples = [{
        'param1': jnp.array(1.0),
        'param2': jnp.array(3.0)
    }]
    assert jnp.array_equal(tree_to_vector(x_samples), jnp.array([[1., 3.]]))
