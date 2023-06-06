from jaxtyping import Array, PyTree
from jax.tree_util import tree_flatten, tree_map
import jax.numpy as jnp

def tree_to_vector(x: PyTree) -> Array:
    x = tree_map(lambda x: x.reshape((x.shape[0], -1)), x)
    x, _ = tree_flatten(x)
    return jnp.concatenate(x, axis=1)
