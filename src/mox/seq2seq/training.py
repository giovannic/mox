import optax
from jaxtyping import Array, PyTree
from flax import linen as nn
from flax.linen.module import _freeze_attr
from typing import Callable, Any
from jax import jit, value_and_grad, random
from jax.tree_util import tree_map
from .rnn import SeqInput
from ..training import training_loss, batch_tree
from ..surrogates import _standardise

def train_seq2seq_surrogate(
        x_in: SeqInput,
        y: PyTree,
        model: nn.Module,
        params: PyTree,
        loss: Callable[[Array, Array], Array],
        key: Any,
        epochs: int = 100,
        n_batches: int = 100,
        optimiser: Any = None
    ) -> PyTree:
    """train_seq2seq_surrogate.
    """
    x_in = _freeze_attr(x_in)

    if optimiser is None:
        tx = optax.adam(learning_rate=.001)
    else:
        tx = optimiser
    opt_state = tx.init(params)

    # standardise y for the loss function
    y = tree_map(_standardise, y, model.y_mean, model.y_std)

    # batch the inputs
    x, x_seq = x_in
    x_batched = batch_tree(x, n_batches)
    x_seq_batched = batch_tree(x_seq, n_batches)
    y_batched = batch_tree(y, n_batches)

    loss_grad_fn = value_and_grad(jit(
        lambda p, x, x_seq, y: training_loss(
            model,
            p,
            loss,
            (x, x_seq),
            y
        )
    ))

    for i in range(epochs):
        key, key_i = random.split(key)

        for b in random.permutation(key_i, n_batches, independent=True):
            loss_val, grads = loss_grad_fn(
                params,
                x_batched[b],
                x_seq_batched[b],
                y_batched[b]
            )
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

    return params
