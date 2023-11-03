from typing import List, Tuple, Optional
from jaxtyping import Array, PyTree
from flax import linen as nn
from flax.linen import Module
import jax.numpy as jnp
from jax import lax
from jax.tree_util import (
    tree_map,
    tree_structure,
    tree_unflatten,
    tree_leaves
)
from .utils import tree_to_vector
from functools import partial

class MLP(Module):
    """MLP. A multi layer perceptron
    """

    units: int
    n_hidden: int
    n_output: int
    dropout_rate: float
    batch_norm: bool

    @nn.compact
    def __call__(self, x, training: bool):
        denses = [nn.Dense(self.units) for _ in range(self.n_hidden)]
        dropouts = [
            nn.Dropout(rate=self.dropout_rate, deterministic=not training)
            for _ in range(self.n_hidden)
        ]
        if self.batch_norm:
            norms = [
                nn.BatchNorm(use_running_average=not training)
                for _ in range(self.n_hidden)
            ]
            layers = zip(denses, dropouts, norms)
            for dense, dropout, norm in layers:
                x = dense(x)
                x = norm(x)
                x = dropout(x)
                x = nn.relu(x)
        else:
            layers = zip(denses, dropouts)
            for dense, dropout in layers:
                x = dense(x)
                x = dropout(x)
                x = nn.relu(x)

        return nn.Dense(self.n_output)(x)

class Surrogate(nn.Module):
    """Surrogate module.

    Surrogate module, which:

    * standardises input arguments
    * flattens the input structure
    * estimates outputs with an nn.Module
    * restructures the outputs
    * applies range restrictions on the the outputs
    """

    x_mean: PyTree
    x_var: PyTree
    y_def: PyTree
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_var: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_hidden: int
    n_output: int
    dropout_rate: float
    batch_norm: bool

    def setup(self):
        self.nn = MLP(
            self.units,
            self.n_hidden,
            self.n_output,
            self.dropout_rate,
            self.batch_norm
        )

    def __call__(self, x, training: bool) -> Array:
        x = self.vectorise(x)

        # predict output
        y = self.nn(x, training)
        return self.recover(y)

    def vectorise(self, x) -> Array:
        # standardise
        x = tree_map(_standardise, x, self.x_mean, self.x_var)

        # vectorise
        x = tree_to_vector(x)
        return x

    def recover(self, y) -> PyTree:
        # recover structure
        y = _recover(y, self.y_boundaries, self.y_shapes, self.y_def)

        # inverse standardise
        y = tree_map(_inverse_standardise, y, self.y_mean, self.y_var)

        # limit outputs
        y = tree_map(
            lambda y, y_min, y_max: maxrelu(minrelu(y, y_min), y_max),
            y,
            self.y_min,
            self.y_max
        )
        return y

def make_surrogate(
        x: list[PyTree],
        y: PyTree,
        x_var_axis: Optional[PyTree] = None,
        y_var_axis: Optional[PyTree] = None,
        y_min: Optional[Array] = None,
        y_max: Optional[Array] = None,
        units = 256,
        n_hidden = 3,
        dropout_rate = .5,
        batch_norm = True
    ) -> nn.Module:
    """make_surrogate.

    Make a surrogate model from a function samples
    """
    x_mean, x_var = safe_summary(summary(x, x_var_axis))
    y_mean, y_var = safe_summary(summary(y, y_var_axis))
    y_shapes = [leaf.shape[1:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])
    n_output = y_boundaries[-1]
    return Surrogate(
        x_mean,
        x_var,
        tree_structure(x_mean),
        y_shapes,
        y_boundaries,
        y_mean,
        y_var,
        y_min,
        y_max,
        units,
        n_hidden,
        n_output,
        dropout_rate,
        batch_norm
    )

def summary(samples: PyTree, axis:PyTree=None) -> Tuple[PyTree, PyTree]:
    """ return mean and variance of a PyTree of samples """
    mean = partial(jnp.mean, keepdims=True)

    if axis is None:
        sample_mean = tree_map(mean, samples)
        return (sample_mean, tree_map(_var, samples, sample_mean))

    sample_mean = tree_map(mean, samples, axis)
    return (
        sample_mean,
        tree_map(_var, samples, sample_mean, axis)
    )

def safe_summary(x: Tuple[PyTree, PyTree]) -> Tuple[PyTree, PyTree]:
    return (x[0], tree_map(lambda leaf: leaf.at[leaf == 0].set(1), x[1]))

def pytree_init(key, model, x):
    return model.init(key, tree_map(lambda x: x[0], x), training=False)

def _var(x, mean, axis=None):
    return jnp.mean(jnp.square(x), axis, keepdims=True) - jnp.square(mean)

def _standardise(x, mu, var):
    return jnp.subtract(x, mu) * lax.rsqrt(var)

def _inverse_standardise(x, mu, var):
    return x * lax.sqrt(var) + mu

def _recover(y, y_boundaries, y_shapes, y_def):
    y_leaves = [
        leaf.reshape(shape)
        for leaf, shape in 
        zip(jnp.split(y, y_boundaries), y_shapes)
    ]
    return tree_unflatten(y_def, y_leaves)

def minrelu(x: Array, min_x: Array) -> Array:
    """minrelu.

    relu with a min input

    :param x: input array
    :type x: Array
    :param min_x: minimum value
    :type min_x: Array
    :rtype: Array
    """
    return jnp.maximum(x, min_x)

def maxrelu(x: Array, max_x: Array) -> Array:
    """minrelu.

    relu with a max input

    :param x: input array
    :type x: Array
    :param max_x: maximum value
    :type max_x: Array
    :rtype: Array
    """
    return jnp.minimum(x, max_x)
