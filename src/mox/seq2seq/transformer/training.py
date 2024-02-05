import optax
from jaxtyping import Array, PyTree
from flax.training import train_state
from flax import linen as nn
from typing import Callable, Any
from jax import jit, value_and_grad, random, vmap, numpy as jnp
from ..rnn import SeqInput, RNNSurrogate
from ...utils import tree_leading_axes as tla
import jax

class TrainState(train_state.TrainState):
    """TrainState for Transformer training."""
    dropout_rng: Array

def train_transformer(
        x_in: SeqInput,
        y: PyTree,
        model: RNNSurrogate,
        net: nn.Module,
        params: PyTree,
        loss_fn: Callable[[PyTree, PyTree], Array],
        key: Any,
        epochs: int = 100,
        batch_size: int = 100,
        optimiser: Any = None,
        vectorising_device = None
    ) -> PyTree:
    """train_transformer

    :param x_in: Input data, a tuple of static and sequential data
    :param y: Output data (sequential)
    :param model: Surrogate model
    :param params: Model parameters
    :param loss_fn: Loss function
    :param key: Random key
    :param epochs: Number of epochs to train for
    :param batch_size: Batch size
    :param optimiser: Optimiser to use, if None, Adam is used
    """
    if optimiser is None:
        tx = optax.adam(learning_rate=.001)
    else:
        tx = optimiser

    with jax.default_device(vectorising_device):
        # standardise x and y
        x = vmap(model.vectorise, in_axes=[tla(x_in)])(x_in)
        y = vmap(model.vectorise_output, in_axes=[tla(y)])(y)

        batches = [
            { 'input': i, 'output': j }
            for i, j
            in zip(jnp.split(x, batch_size), jnp.split(y, batch_size))
        ]

    n_batches = len(batches)

    key, dropout_key = random.split(key)

    state = TrainState.create(
        dropout_rng=dropout_key,
        apply_fn=net.apply,
        params=params,
        tx=tx
    )

    @jit
    def train_step(
        state: TrainState,
        batch: PyTree
        ):
        dropout_train_key = random.fold_in(
            key=state.dropout_rng,
            data=state.step
        )

        def apply_loss(params):
            estimate = state.apply_fn(
                params,
                batch['input'],
                train=True,
                rngs={'dropout': dropout_train_key}
            )
            loss = loss_fn(estimate, batch['output'])
            return loss

        grad_fn = value_and_grad(apply_loss)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = { 'loss': loss }
        return state, metrics

    for _ in range(epochs):
        key, key_i = random.split(key)

        for b in random.permutation(key_i, n_batches, independent=True):
            state, _ = train_step(state, batches[b])

    return state