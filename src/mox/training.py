import optax
from jax import jit, value_and_grad, random, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import PyTree, Array
from typing import Any, Callable
from flax import linen as nn
from flax.training import train_state
from .utils import tree_leading_axes as tla
from .surrogates import Surrogate

# TODO:
# * Update train_surrogate documentation
# * Make real test for train_surrogate
# * Fix evaluate_surrogate and test
# * Remove l2 loss
# * Reorganise Surrogate module to not use PyTrees and Flax at the same time

class TrainState(train_state.TrainState):
    batch_stats: Any

def batched_training_loss(
    model: nn.Module,
    params: PyTree,
    loss: Callable[[PyTree, PyTree], Array],
    x: PyTree,
    y: PyTree
    ) -> Array:
    return jnp.mean(
        vmap(
            lambda x, y: training_loss(model, params, loss, x, y),
            in_axes=[tree_map(lambda _: 0, x), tree_map(lambda _: 0, y)]
        )(
            x,
            y
        ),
        axis=0
    )

def training_loss(
    model: nn.Module,
    params: PyTree,
    loss: Callable[[PyTree, PyTree], Array],
    x: PyTree,
    y: PyTree
    ) -> Array:
    y_hat = model.apply(
        params,
        x,
        method = lambda module, x: module.unstandardised(x)
    )
    return loss(y, y_hat)

def train_surrogate(
        x: list[PyTree],
        y: PyTree,
        model: Surrogate,
        net: nn.Module,
        loss_fn: Callable[[PyTree, PyTree], Array],
        key: Any,
        variables: PyTree,
        epochs: int = 100,
        batch_size: int = 100,
        optimiser: Any = None
    ) -> TrainState:
    """train_surrogate.
    
    Train a surrogate module on samples x and y

    :param x: Function parameter samples
    :type x: list[PyTree]
    :param y: Function outputs
    :type y: Array
    :param loss: Loss function for training
    :type loss: Callable[[Array, Array], float]
    :rtype: nn.Module
    """
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})

    if optimiser is None:
        tx = optax.adam(learning_rate=.001)
    else:
        tx = optimiser

    # standardise x and y
    x_vec = vmap(model.vectorise, in_axes=[tla(x)])(x)
    y_vec = vmap(model.vectorise_output, in_axes=[tla(y)])(y)

    batches = [
        { 'input': i, 'output': j }
        for i, j
        in zip(jnp.split(x_vec, batch_size), jnp.split(y_vec, batch_size))
    ]

    n_batches = len(batches)

    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx
    )

    @jit
    def train_step(state: TrainState, batch):
        dropout_key = random.fold_in(key, state.step)

        def apply_loss(params):
            estimate, updates = state.apply_fn(
                {
                    'params': params,
                    'batch_stats': state.batch_stats
                },
                batch['input'],
                True,
                rngs={ 'dropout': dropout_key },
                mutable=['batch_stats']
            )
            loss = loss_fn(estimate, batch['output'])
            return loss, updates

        grad_fn = value_and_grad(apply_loss, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
        metrics = { 'loss': loss }
        return state, metrics

    for _ in range(epochs):
        key, key_i = random.split(key)

        for b in random.permutation(key_i, n_batches, independent=True):
            state, _ = train_step(state, batches[b])

    return state

def evaluate_surrogate(
        x: list[PyTree],
        y: PyTree,
        state: TrainState,
        loss_fn: Callable[[PyTree, PyTree], Array],
        ):

    @jit
    def eval_state(state: TrainState, batch):
        estimate = state.apply_fn(
            {
                'params': state.params,
                'batch_stats': state.batch_stats
            },
            batch['input'],
            method=lambda module, x: module.nn(x, False)
        )
        loss = loss_fn(estimate, batch['output'])
        metrics = { 'loss': loss }
        return state, metrics

    return eval_state(state, { 'input': x, 'output': y })
