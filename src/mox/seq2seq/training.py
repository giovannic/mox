import optax
from jaxtyping import Array, PyTree
from flax.linen.module import _freeze_attr
from flax.training import train_state
from typing import Callable, Any
from jax import jit, value_and_grad, random, vmap, numpy as jnp
from jax.tree_util import tree_map
from .rnn import SeqInput, RNNSurrogate

def train_rnn_surrogate(
        x_in: SeqInput,
        y: PyTree,
        model: RNNSurrogate,
        params: PyTree,
        loss_fn: Callable[[PyTree, PyTree], Array],
        key: Any,
        epochs: int = 100,
        batch_size: int = 100,
        optimiser: Any = None
    ) -> PyTree:
    """train_seq2seq_surrogate.
    """
    x_in = _freeze_attr(x_in)

    if optimiser is None:
        tx = optax.adam(learning_rate=.001)
    else:
        tx = optimiser

    # standardise x and y
    x = vmap(model.vectorise, in_axes=[tree_map(lambda _: 0, x_in)])(x_in)
    y = vmap(model.vectorise_output, in_axes=[tree_map(lambda _: 0, y)])(y)

    batches = [
        { 'input': i, 'output': j }
        for i, j
        in zip(jnp.split(x, batch_size), jnp.split(y, batch_size))
    ]

    n_batches = len(batches)

    state = train_state.TrainState.create(
        apply_fn=model.net.apply,
        params=params,
        tx=tx
    )

    @jit
    def train_step(state: train_state.TrainState, batch: PyTree):
        def apply_loss(params):
            estimate = state.apply_fn(
                params,
                batch['input'],
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
