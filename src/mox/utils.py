from jaxtyping import Array, PyTree
from jax.tree_util import (
    tree_leaves,
    tree_map,
    register_pytree_node,
    tree_flatten
)
import jax.numpy as jnp
import dataclasses

def tree_leading_axes(x: PyTree) -> PyTree:
    return tree_map(lambda _: 0, x)

def tree_to_vector(x: PyTree) -> Array:
    return jnp.concatenate([jnp.ravel(x) for x in tree_leaves(x)])

def unbatch_tree(x: PyTree) -> PyTree:
    return tree_map(lambda x: x[0], x)

def register_dataclass_as_pytree(cls):
    register_pytree_node(
        cls,
        lambda o: tree_flatten(dataclasses.asdict(o)),
        lambda d, leaves: cls(**d.unflatten(leaves))
    )
    return cls
