# Adapted from flax seq2seq examples:
# https://github.com/google/flax/blob/main/examples/seq2seq/

from typing import Tuple, Optional, List, Union
from jaxtyping import Array, PyTree
from flax import linen as nn
from jax.random import KeyArray
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_structure, tree_map
from jax import vmap

from ..surrogates import (
    Vectoriser,
    Standardiser,
    InverseStandardiser,
    Limiter,
    summary
)

from .encoding import (FillEncoding, filler)
from .surrogates import RecoverSeq

PRNGKey = KeyArray
LSTMCarry = Tuple[Array, Array]
SeqInput = Tuple[List[PyTree], List[PyTree]]

class SequenceVectoriser(nn.Module):

    @nn.compact
    def __call__(self, x: PyTree):
        return jnp.concatenate([
            x.reshape(x.shape[:2] + (-1,))
            for x in tree_leaves(x)
        ], axis=2)

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
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_output: Union[int, Array]

    def setup(self):
        # static encoding
        self.std = Standardiser(self.x_mean, self.x_std)
        self.vec = Vectoriser()

        # sequence translation
        self.std_seq = Standardiser(self.x_seq_mean, self.x_seq_std)
        self.vec_seq = SequenceVectoriser()
        self.filler = FillEncoding(self.filler_pattern)
        self.rnn = nn.RNN(DecoderLSTMCell(self.units, self.n_output))

        # post prediction recovery
        self.rec = RecoverSeq(
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries
        )
        self.inv_std = InverseStandardiser(self.y_mean, self.y_std)
        if self.y_min is not None:
            self.limiter = Limiter(self.y_min, self.y_max)
        else:
            self.limiter = lambda x: x

    def __call__(self, x_in: SeqInput):
        y = self.limiter(self.inv_std(self.unstandardised(x_in)))
        return y

    def unstandardised(self, x_in:SeqInput):
        x, x_seq = x_in
        # encode static
        x = self.std(x)
        x_vec = vmap(self.vec, in_axes=[tree_map(lambda _: 0, x)])(x)

        # encode sequence
        x_seq = self.std_seq(x_seq)
        x_seq = tree_map(
            lambda leaf: vmap(self.filler)(leaf),
            x_seq
        )
        x_seq_vec = self.vec_seq(x_seq)

        # jointly decode sequence
        x_rep = jnp.repeat(
            x_vec[:, jnp.newaxis, :],
            self.n_steps,
            axis=1
        )
        x_joint = jnp.concatenate([x_rep, x_seq_vec], axis=2)
        y = self.rnn(x_joint)
        y = vmap(self.rec, in_axes=[0, None])(y, self.n_steps)
        return y

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

    x_mean, x_std = summary(x, x_std_axis)
    x_seq_mean, x_seq_std = summary(x_seq, x_seq_std_axis)
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
        filler(x_t, n_steps),
        y_shapes,
        y_boundaries,
        y_mean,
        y_std,
        y_min,
        y_max,
        units,
        n_output
    )
