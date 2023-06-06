import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_leaves

def assert_tree_equal(x, y):
    assert tree_structure(x) == tree_structure(y)
    assert all(
        jnp.array_equal(lx, ly)
        for lx, ly in zip(tree_leaves(x), tree_leaves(y))
    )
