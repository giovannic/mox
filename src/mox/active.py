import optax
from typing import Callable, Any, List
from jaxtyping import Array, PyTree
from flax import linen as nn
from flax.linen.module import _freeze_attr
from jax.tree_util import tree_map
from jax import jit, value_and_grad, random, vmap, grad
from jax import numpy as jnp
from .training import training_loss, batch_tree
from .sampling import sample, ParamStrategy
from .surrogates import minrelu, maxrelu

def active_training(
        x: list[PyTree],
        y: PyTree,
        model: nn.Module,
        params: PyTree,
        loss: Callable[[Array, Array], Array],
        key: Any,
        f: Callable,
        utility: Callable,
        pool_strategy: List[ParamStrategy],
        epochs: int = 100,
        batch_size: int = 100,
        learning_rate: float = .001,
        n_pool: int = 100,
        pool_iter: int = 100,
        replace_pool: bool = True,
        x_min: PyTree = None,
        x_max: PyTree = None,
        pool_learning_rate: float = .01
    ):
    x = _freeze_attr(x)

    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(params)
    loss_grad_fn = value_and_grad(jit(
        lambda p, x, y: training_loss(model, p, loss, x, y)
    ))

    x_train = x
    y_train = y

    for i in range(epochs):
        key, key_i = random.split(key)
        x_pool = _freeze_attr(sample(pool_strategy, n_pool, key_i))
        x_pool = sample_towards_utility(
            x_pool,
            model,
            params,
            utility,
            x_min,
            x_max,
            pool_iter,
            pool_learning_rate
        )
        y_pool = vmap(f, in_axes=tree_map(lambda _: 0, x_pool))(*x_pool)

        if replace_pool:
            x_train = tree_map(concat, x, _freeze_attr(x_pool))
            y_train = tree_map(concat, y, _freeze_attr(y_pool))
        else:
            x_train = tree_map(concat, x_train, _freeze_attr(x_pool))
            y_train = tree_map(concat, y_train, _freeze_attr(y_pool))

        x_train_batched = batch_tree(x_train, batch_size)
        y_train_batched = batch_tree(y_train, batch_size)
        n_batches = len(x)

        for b in random.permutation(key_i, n_batches, independent=True):
            loss_val, grads = loss_grad_fn(
                params,
                x_train_batched[b],
                y_train_batched[b]
            )
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

    return params

def sample_towards_utility(
        x: PyTree,
        model: nn.Module,
        params: PyTree,
        utility: Callable,
        x_min: PyTree = None,
        x_max: PyTree = None,
        epochs: int = 100,
        learning_rate: float = .01
    ):
    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(x)
    loss_fn = jit(grad(lambda x: utility(x, model, params)))

    for _ in range(epochs):
        grads = loss_fn(x)
        updates, opt_state = optimiser.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)
        if x_min is not None:
            x = tree_map(
                lambda x, x_min: minrelu(x, x_min),
                x,
                x_min
            )
        if x_max is not None:
            x = tree_map(
                lambda x, x_max: maxrelu(x, x_max),
                x,
                x_max
            )

    return x


def concat(a, b): 
    return jnp.concatenate([a, b])
