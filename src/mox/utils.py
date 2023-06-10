from jaxtyping import Array, PyTree
from jax.tree_util import tree_leaves
import jax.numpy as jnp

def tree_to_vector(x: PyTree) -> Array:
    return jnp.concatenate([jnp.ravel(x) for x in tree_leaves(x)])
