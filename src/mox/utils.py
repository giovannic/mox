from jaxtyping import Array, PyTree
from jax.tree_util import tree_leaves, tree_map
import jax.numpy as jnp

def tree_leading_axes(x: PyTree) -> PyTree:
    return tree_map(lambda _: 0, x)

def tree_to_vector(x: PyTree) -> Array:
    return jnp.concatenate([jnp.ravel(x) for x in tree_leaves(x)])

def unbatch_tree(x: PyTree) -> PyTree:
    return tree_map(lambda x: x[0], x)
