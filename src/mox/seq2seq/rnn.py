# Adapted from flax seq2seq examples:
# https://github.com/google/flax/blob/main/examples/seq2seq/

from typing import Tuple, Optional, List, Any
from jaxtyping import Array, PyTree
from flax import linen as nn
from jax.random import KeyArray
import jax.numpy as jnp
from jax.tree_util import (
    tree_leaves,
    tree_structure,
    tree_map,
    tree_unflatten
)

from ..surrogates import (
    summary,
    safe_summary,
    minrelu,
    maxrelu,
    _standardise,
    _inverse_standardise
)

from ..utils import tree_to_vector

PRNGKey = KeyArray
LSTMCarry = Tuple[Array, Array]
SeqInput = Tuple[List[PyTree], List[PyTree]]

class DecoderLSTMCell(nn.RNNCellBase):
    """DecoderLSTM Module wrapped in a lifted scan transform.
    feature_size: Feature size of the output sequence
    """
    units: int
    feature_size: int

    def setup(self):
        self.lstm = nn.LSTMCell(self.units)
        self.dense = nn.Dense(features=self.feature_size)

    def __call__(
          self,
          carry: LSTMCarry,
          x: Array
          ) -> Tuple[LSTMCarry, Array]:
        """Applies the DecoderLSTM model."""
        carry, y = self.lstm(carry, x)
        prediction = self.dense(y)
        return carry, prediction

    def initialize_carry(self, rng, input_shape) -> LSTMCarry:
        return self.lstm.initialize_carry(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        return 1

class RNNSurrogate(nn.Module):
    n_steps: Array
    x_mean: PyTree
    x_std: PyTree
    x_seq_mean: PyTree
    x_seq_std: PyTree
    filler_pattern: Array
    y_def: PyTree
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_output: int

    def setup(self):
        # sequence translation
        self.rnn = nn.RNN(DecoderLSTMCell(self.units, self.n_output))

    def __call__(self, x: SeqInput) -> Array:
        x_vec = self.vectorise(x)
        y = self.rnn(x_vec)
        return self.recover(y)

    def vectorise(self, x: SeqInput) -> Array:
        x_static, x_seq = x
        # standardise
        x_static = tree_map(_standardise, x_static, self.x_mean, self.x_std)
        x_seq = tree_map(_standardise, x_seq, self.x_seq_mean, self.x_seq_std)

        # fill sequence
        x_seq = _fill(x_seq, self.filler_pattern, self.n_steps)

        # vectorise
        x_static = tree_to_vector(x_static)
        x_seq = _vectorise_sequence(x_seq)

        x_rep = jnp.repeat(
            x_static[:, jnp.newaxis, :],
            self.n_steps,
            axis=1
        )
        return jnp.concatenate([x_rep, x_seq], axis=2)

    def recover(self, y: Any) -> PyTree:
        y = tree_map(_inverse_standardise, y, self.y_mean, self.y_std)
        y_tree = _recover_sequence(
            y,
            self.y_shapes,
            self.y_def,
            self.y_boundaries,
            self.n_steps
        )

        # limit outputs
        y_tree = tree_map(
            lambda y, y_min, y_max: maxrelu(minrelu(y, y_min), y_max),
            y,
            self.y_min,
            self.y_max
        )
        return y_tree

def make_rnn_surrogate(
    x: list[PyTree],
    x_seq: list[PyTree],
    x_t: Array,
    n_steps: Array,
    y: PyTree,
    x_std_axis: Optional[PyTree] = None,
    x_seq_std_axis: Optional[PyTree] = None,
    y_std_axis: Optional[PyTree] = None,
    y_min: Optional[Array] = None,
    y_max: Optional[Array] = None,
    units = 256,
    ):

    x_mean, x_std = safe_summary(summary(x, x_std_axis))
    x_seq_mean, x_seq_std = safe_summary(summary(x_seq, x_seq_std_axis))
    y_mean, y_std = summary(y, y_std_axis)
    y_shapes = [leaf.shape[2:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])
    n_output = y_boundaries[-1]
    return RNNSurrogate(
        n_steps,
        x_mean,
        x_std,
        x_seq_mean,
        x_seq_std,
        _filler(x_t, n_steps),
        tree_structure(y_mean),
        y_shapes,
        y_boundaries,
        y_mean,
        y_std,
        y_min,
        y_max,
        units,
        n_output
    )

def _vectorise_sequence(x: PyTree) -> Array:
    return jnp.concatenate([
        x.reshape(x.shape[:2] + (-1,))
        for x in tree_leaves(x)
    ], axis=2)

def _recover_sequence(
    y: Array,
    y_shapes: List[Tuple],
    y_def: PyTree,
    y_boundaries: Tuple,
    n_steps: Array
    ) -> PyTree:
    y_leaves = [
        leaf.reshape(y.shape[:1] + shape)[:n_steps]
        for leaf, shape in 
        zip(jnp.split(y, y_boundaries[:-1], axis=1), y_shapes)
    ]
    return tree_unflatten(y_def, y_leaves)

def _filler(t, max_t):
    return jnp.diff(jnp.append(t, max_t))

def _fill(x: Array, pattern: Array, steps: Array):
    return jnp.repeat(x, pattern, axis=0)[:steps]
