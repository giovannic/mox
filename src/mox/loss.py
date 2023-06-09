from jax import numpy as jnp
from jax.nn import softplus
from jaxtyping import PyTree, Array
from jax.tree_util import tree_leaves

def _diffs(x: PyTree, y: PyTree) -> Array:
    return jnp.concatenate([
        (x_leaf - y_leaf).reshape(-1)
        for x_leaf, y_leaf in zip(tree_leaves(x), tree_leaves(y))
    ])

def mse(x: PyTree, y: PyTree) -> float:
    return jnp.mean(jnp.square(_diffs(x, y)))

def log_cosh(x: PyTree, y: PyTree) -> float:
    diff = _diffs(x, y)
    return jnp.mean(diff + softplus(-2. * diff) - jnp.log(2.))
