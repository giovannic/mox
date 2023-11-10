import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_leaves
from jaxtyping import Array, PyTree
from flax.linen.recurrent import _Never

def _is_never(x, y) -> bool:
    return isinstance(x, _Never) and isinstance(y, _Never)

def assert_tree_equal(x: PyTree, y: PyTree):
    assert tree_structure(x) == tree_structure(y)
    assert all(
        _is_never(lx, ly) or jnp.all(lx == ly)
        for lx, ly in zip(tree_leaves(x), tree_leaves(y))
    )

def assert_tree_roughly_equal(x: PyTree, y: PyTree, epsilon=1e-5):
    assert tree_structure(x) == tree_structure(y)
    assert all(
        roughly_equal(lx, ly, epsilon=epsilon)
        for lx, ly in zip(tree_leaves(x), tree_leaves(y))
    )

def roughly_equal(x: Array, y: Array, epsilon=1e-5) -> bool:
    return jnp.all(jnp.abs(x - y) < epsilon)
