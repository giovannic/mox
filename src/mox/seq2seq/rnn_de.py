# Adapted from flax seq2seq examples:
# https://github.com/google/flax/blob/main/examples/seq2seq/

from typing import Tuple, Optional, List, Union
from jaxtyping import Array, PyTree
from flax import linen as nn
from jax.random import KeyArray
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_structure, tree_map

from ..surrogates import (
    Vectoriser,
    Standardiser,
    Recover,
    InverseStandardiser,
    Limiter,
    summary
)

from .encoding import PositionalEncoding

PRNGKey = KeyArray
LSTMCarry = Tuple[Array, Array]

class SequenceVectoriser(nn.Module):

    max_t: int
    eos_id: float

    @nn.compact
    def __call__(self, x: PyTree):
        return jnp.concatenate([
            jnp.pad(
                x.reshape(x.shape[:2] + (-1,)),
                ((0, 0), (0, self.max_t - x.shape[1]), (0, 0)),
                constant_values=self.eos_id
            )
            for x in tree_leaves(x)
        ], axis=2)


class DecoderLSTMCell(nn.RNNCellBase):
    """DecoderLSTM Module wrapped in a lifted scan transform.

    Attributes:
    teacher_force: See docstring on Seq2seq module.
    feature_size: Feature size of the output sequence
    """
    teacher_force: bool
    feature_size: int

    @nn.compact
    def __call__(
          self,
          carry: Tuple[LSTMCarry, Array],
          x: Array
          ) -> Tuple[Tuple[LSTMCarry, Array], Array]:
        """Applies the DecoderLSTM model."""
        lstm_state, last_prediction = carry
        if not self.teacher_force:
            x = last_prediction
        lstm_state, y = nn.LSTMCell()(lstm_state, x)
        prediction = nn.Dense(features=self.feature_size)(y)
        carry = (lstm_state, prediction)
        return carry, prediction

class RNNDESurrogate(nn.Module):
    x_t: Array
    x_mean: PyTree
    x_std: PyTree
    x_seq_mean: PyTree
    x_seq_std: PyTree
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_output: Union[int, Array]
    max_t: int
    eos_id: float
    teacher_force: bool = True

    def setup(self):
        # static encoding
        self.std = Standardiser(self.x_mean, self.x_std)
        self.vec = Vectoriser()
        self.dense = nn.Dense(self.units)

        # sequence encoding
        self.std_seq = Standardiser(self.x_seq_mean, self.x_seq_std)
        self.vec_seq = SequenceVectoriser(self.max_t, self.eos_id)
        self.dense_seq = nn.Dense(self.units)
        self.pos_enc = PositionalEncoding(self.units, self.max_t)
        self.encoder = nn.RNN(
            nn.LSTMCell(),
            self.units,
            return_carry=True
        )

        # joint decoding
        self.decoder = nn.RNN(
            DecoderLSTMCell(
                self.teacher_force,
                self.n_output
            ),
            self.n_output
        )

        # post prediction recovery
        self.rec = Recover(
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries
        )
        self.inv_std = InverseStandardiser(self.y_mean, self.y_std)
        if self.y_min is not None:
            self.limiter = Limiter(self.y_min, self.y_max)
        else:
            self.limiter = lambda x: x

    def __call__(self, x, x_seq, x_t, y):
        y = self.limiter(self.inv_std(self.unstandardised(x, x_seq, x_t, y)))
        return y

    def unstandardised(self, x, x_seq, x_t, y):
        # encode static
        x = self.std(x)
        x = self.vec(x)
        x = self.dense(x)

        # encode sequence
        seq_lengths = self.get_seq_lengths(x_seq)
        x_seq = self.std_seq(x_seq)
        x_seq = tree_map(lambda leaf: self.pos_enc(leaf, x_t), x_seq)
        x_seq = self.vec_seq(x_seq)
        x_seq = self.dense_seq(x_seq)
        x_seq_enc, _ = self.encoder(x_seq, seq_lengths=seq_lengths)

        # jointly decode sequence
        x_joint = jnp.concatenate([x, x_seq_enc])
        init_pred = jnp.full((self.n_output,), -1)
        y = self.decoder(
            y,
            initial_carry=(x_joint, init_pred)
        )
        y = self.rec(y)
        return y

    def get_seq_lengths(self, inputs: PyTree) -> PyTree:
        """Get segmentation mask for inputs."""
        return tree_leaves(inputs)[0].shape[1]

def make_rnn_de_surrogate(
    x: list[PyTree],
    x_seq: list[PyTree],
    x_t: Array,
    y: PyTree,
    x_std_axis: Optional[PyTree] = None,
    x_seq_std_axis: Optional[PyTree] = None,
    y_std_axis: Optional[PyTree] = None,
    y_min: Optional[Array] = None,
    y_max: Optional[Array] = None,
    units = 256,
    max_t = 100
    ):

    x_mean, x_std = summary(x, x_std_axis)
    x_seq_mean, x_seq_std = summary(x_seq, x_seq_std_axis)
    y_mean, y_std = summary(y, y_std_axis)
    y_shapes = [leaf.shape[1:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])
    n_output = y_boundaries[-1]
    return RNNDESurrogate(
        x_t,
        x_mean,
        x_std,
        x_seq_mean,
        x_seq_std,
        y_shapes,
        y_boundaries,
        y_mean,
        y_std,
        y_min,
        y_max,
        units,
        n_output,
        max_t,
        -1
    )
