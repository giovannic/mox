from typing import Callable, Tuple
from jax import numpy as jnp, vmap
from jax.nn import softplus
from jaxtyping import PyTree, Array
from jax.tree_util import (
    tree_leaves,
    tree_map,
    tree_map_with_path,
    tree_flatten
)
import flax.linen as nn

LossSignature = Callable[[nn.Module, PyTree, PyTree, PyTree], Array]

def mse(x: Array, y: Array) -> Array:
    return jnp.mean(jnp.square(_diffs(x, y)))

def log_cosh(x: PyTree, y: PyTree) -> Array:
    diff = _diffs(x, y)
    return jnp.mean(diff + softplus(-2. * diff) - jnp.log(2.))

def l2_loss(params: PyTree, alpha: Array):
    return jnp.sum(
        alpha * jnp.stack(
            tree_flatten(
                tree_map_with_path(_l2_loss_leaf, params)
            )[0]
        )
    )

def make_predictive_loss(f: Callable[[PyTree, PyTree], Array]) -> LossSignature:
    def loss(model: nn.Module, params: PyTree, x: PyTree, y: PyTree) -> Array:
        return jnp.mean(
            vmap(
                lambda x, y: standardised_loss(model, params, f, x, y),
                in_axes=[tree_map(lambda _: 0, x), tree_map(lambda _: 0, y)]
            )(
                x,
                y
            ),
            axis=0
        )
    return loss

def make_regularised_predictive_loss(
    f: Callable[[PyTree, PyTree], Array],
    alpha: Array
    ) -> LossSignature:

    def loss(model: nn.Module, params: PyTree, x: PyTree, y: PyTree) -> Array:
        return jnp.mean(
            vmap(
                lambda x, y: standardised_loss(model, params, f, x, y),
                in_axes=[tree_map(lambda _: 0, x), tree_map(lambda _: 0, y)]
            )(
                x,
                y
            ),
            axis=0
        ) + l2_loss(params, alpha)
    return loss

def standardised_loss(
    model: nn.Module,
    params: PyTree,
    loss: Callable[[PyTree, PyTree], Array],
    x: PyTree,
    y: PyTree
    ) -> Array:
    y_hat = model.apply(
        params,
        x,
        method = lambda module, x: module.standardised(x)
    )
    return loss(y, y_hat)

def _l2_loss_leaf(path: Tuple, leaf: Array) -> Array:
    if path[-1].key == 'kernel':
        return (leaf ** 2).mean()
    return jnp.array(0.)

def _diffs(x: PyTree, y: PyTree) -> Array:
    return jnp.concatenate([
        (x_leaf - y_leaf).reshape(-1)
        for x_leaf, y_leaf in zip(tree_leaves(x), tree_leaves(y))
    ])
