from typing import Tuple, Optional, List, Any
from jaxtyping import Array, PyTree
from flax import linen as nn
from jax import numpy as jnp, vmap
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

from ..utils import (
    tree_to_vector,
    unbatch_tree,
    tree_leading_axes as tla,
    register_dataclass_as_pytree
)

from dataclasses import dataclass

PRNGKey = Array
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

@register_dataclass_as_pytree
@dataclass
class RNNSurrogate():
    n_steps: Array
    x_mean: PyTree
    x_var: PyTree
    x_seq_mean: PyTree
    x_seq_var: PyTree
    filler_pattern: Array
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_var: PyTree
    y_min: PyTree
    y_max: PyTree

    def __call__(self, x: SeqInput, net: nn.Module, params: PyTree) -> Array:
        x_vec = self.vectorise(x)
        y = net.apply(params, x_vec)
        return self.recover(y)

    def vectorise(self, x: SeqInput) -> Array:
        x_static, x_seq = x
        # standardise
        x_static = unbatch_tree(tree_map(_standardise, x_static, self.x_mean, self.x_var))
        x_seq = unbatch_tree(tree_map(_standardise, x_seq, self.x_seq_mean, self.x_seq_var))

        # vectorise
        x_static = tree_to_vector(x_static)
        x_seq = _vectorise_sequence(x_seq)

        # fill sequence
        x_seq = _fill(x_seq, self.filler_pattern, self.n_steps)

        x_rep = jnp.repeat(
            x_static[jnp.newaxis, :],
            self.n_steps,
            axis=0
        )
        return jnp.concatenate([x_rep, x_seq], axis=1)

    def vectorise_output(self, y: PyTree) -> Array:
        # standardise
        y = unbatch_tree(tree_map(_standardise, y, self.y_mean, self.y_var))

        # vectorise
        return _vectorise_sequence(y)

    def get_output_dim(self, y: PyTree) -> int:
        vec = vmap(self.vectorise_output, in_axes=[tla(y)])(y)
        return vec.shape[-1]

    def recover(self, y: Any) -> PyTree:
        y = _recover_sequence(
            y,
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries,
            self.n_steps
        )
        y = unbatch_tree(
            tree_map(_inverse_standardise, y, self.y_mean, self.y_var)
        )

        # limit outputs
        if self.y_min is not None:
            y = tree_map(
                lambda y, y_min: minrelu(y, y_min),
                y,
                self.y_min
            )

        if self.y_max is not None:
            y = tree_map(
                lambda y, y_max: maxrelu(y, y_max),
                y,
                self.y_max
            )

        return y

class RNNDensitySurrogate(RNNSurrogate):

    def recover(self, y: Any) -> PyTree:
        mu, log_sigma = y
        mu = _recover_sequence(
            mu,
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries,
            self.n_steps
        )

        mu = unbatch_tree(
            tree_map(_inverse_standardise, mu, self.y_mean, self.y_var)
        )

        if self.y_min is not None:
            mu = tree_map(
                lambda y, y_min: minrelu(y, y_min),
                mu,
                self.y_min
            )

        if self.y_max is not None:
            mu = tree_map(
                lambda y, y_max: maxrelu(y, y_max),
                mu,
                self.y_max
            )

        log_sigma = _recover_sequence(
            log_sigma,
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries,
            self.n_steps
        )

        log_sigma = unbatch_tree(
            tree_map(
                lambda leaf, y_var: leaf + .5 * jnp.log(y_var),
                log_sigma,
                self.y_var
            )
        )

        return mu, log_sigma

def make_rnn_surrogate(
    x: list[PyTree],
    x_seq: list[PyTree],
    x_t: Array,
    n_steps: Array,
    y: PyTree,
    x_var_axis: Optional[PyTree] = None,
    x_seq_var_axis: Optional[PyTree] = None,
    y_var_axis: Optional[PyTree] = None,
    y_min: Optional[Array] = None,
    y_max: Optional[Array] = None,
    density: bool = False
    ):
    if x_seq_var_axis is None:
        x_seq_var_axis = default_aggregation_axes(x_seq)
    if y_var_axis is None:
        y_var_axis = default_aggregation_axes(y_var_axis)

    x_mean, x_var = safe_summary(summary(x, x_var_axis))
    x_seq_mean, x_seq_var = safe_summary(summary(x_seq, x_seq_var_axis))
    y_mean, y_var = safe_summary(summary(y, y_var_axis))
    y_shapes = [leaf.shape[2:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])

    if density:
        return RNNDensitySurrogate(
            n_steps,
            x_mean,
            x_var,
            x_seq_mean,
            x_seq_var,
            _filler(x_t, n_steps),
            y_shapes,
            y_boundaries,
            y_mean,
            y_var,
            y_min,
            y_max
        )
    return RNNSurrogate(
        n_steps,
        x_mean,
        x_var,
        x_seq_mean,
        x_seq_var,
        _filler(x_t, n_steps),
        y_shapes,
        y_boundaries,
        y_mean,
        y_var,
        y_min,
        y_max
    )

def _vectorise_sequence(x: PyTree) -> Array:
    return jnp.concatenate([
        leaf.reshape((leaf.shape[0], -1))
        for leaf in tree_leaves(x)
    ], axis=1)

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

def default_aggregation_axes(x: PyTree) -> PyTree:
    """
    For a given sequential pytree, return the axes to aggregate over
    """
    return tree_map(lambda _: (0, 1), x)

def init_surrogate(key, surrogate: RNNSurrogate, net: nn.Module, x):
    x_vec = vmap(surrogate.vectorise, in_axes=[tla(x)])(x)
    return net.init(key, x_vec)

def apply_surrogate(surrogate: RNNSurrogate, net: nn.Module, params: PyTree, x):
    x_vec = vmap(surrogate.vectorise, in_axes=[tla(x)])(x)
    y = net.apply(params, x_vec)
    return vmap(surrogate.recover)(y)
