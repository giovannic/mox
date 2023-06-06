from typing import Sequence, List, Tuple
from jaxtyping import Array, PyTree
from flax import linen as nn
import jax.numpy as jnp
from jax.tree_util import (
    tree_map,
    tree_structure,
    tree_unflatten,
    tree_leaves
)
from .utils import tree_to_vector

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

class MLP(nn.Module):
    """MLP. A multi layer perceptron
    """

    units: int
    n_hidden: int
    n_output: int

    @nn.compact
    def __call__(self, x):
        layers = [nn.Dense(self.units) for _ in range(self.n_hidden)]
        for i, lyr in enumerate(layers):
            x = lyr(x)
            x = nn.relu(x)
        return nn.Dense(self.n_output)(x)

class Vectoriser(nn.Module):

    x_mean: Sequence[PyTree]
    x_std: Sequence[PyTree]

    @nn.compact
    def __call__(self, x):
        x = self.standardise(x)
        return tree_to_vector(x)

    def standardise(self, x):
        return tree_map(
            _standardise,
            x,
            self.x_mean,
            self.x_std
        )

class Recover(nn.Module):
    """Recover. Recover output PyTree from vectorised neural net output"""

    y_shapes: List[Array]
    y_mean: PyTree
    y_std: PyTree

    @nn.compact
    def __call__(self, y):
        y_boundaries = jnp.cumsum(
            jnp.array([jnp.prod(s) for s in self.y_shapes])
        )
        y_leaves = [
            leaf.reshape((y.shape[0],) + tuple(shape))
            for leaf, shape in 
            zip(jnp.split(y, y_boundaries, axis=1), self.y_shapes)
        ]
        y = tree_unflatten(
            tree_structure(self.y_mean),
            y_leaves
        )
        return self.recover(y)

    def recover(self, y: Array):
        """recover. Perform inverse standardisation on y

        :param y: 
        """
        return tree_map(
            _inverse_standardise,
            y,
            self.y_mean,
            self.y_std
        )

class Limiter(nn.Module):
    """Limiter. limit outputs using relus"""

    y_min: Array
    y_max: Array

    @nn.compact
    def __call__(self, y):
        return maxrelu(minrelu(y, self.y_min), self.y_max)

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
    x_std: PyTree
    y_shapes: List[Array]
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_hidden: int
    n_output: int

    def setup(self):
        self.vec = Vectoriser(self.x_mean, self.x_std)
        self.rec = Recover(
            self.y_shapes,
            self.y_mean,
            self.y_std,
            self.y_min,
            self.y_max
        )
        self.nn = MLP(self.units, self.n_hidden, self.n_output)
        if self.y_min is not None:
            self.limiter = Limiter(
                _standardise(tree_to_vector(self.y_min), self.y_mean, self.y_std),
                _standardise(tree_to_vector(self.y_max), self.y_mean, self.y_std)
            )
        else:
            self.limiter = lambda x: x

    def __call__(self, x):
        x = self.vec(x)
        y = self.nn(x)
        y = self.limiter(y)
        return self.rec(y)

def make_surrogate(
        x: list[PyTree],
        y: PyTree,
        nn: nn.Module=MLP,
        x_std_axis: PyTree = None,
        y_std_axis: PyTree = None,
        y_min: Array = None,
        y_max: Array = None,
        units = 256,
        n_hidden = 3
    ) -> nn.Module:
    """make_surrogate.

    Make a surrogate model from a function samples
    """
    x_mean, x_std = summary(x, x_std_axis)
    y_mean, y_std = summary(y, y_std_axis)
    y_shapes = [jnp.array(leaf.shape[1:]) for leaf in tree_leaves(y)]
    n_output = sum(jnp.prod(shape) for shape in y_shapes)
    return Surrogate(
        x_mean,
        x_std,
        y_shapes,
        y_mean,
        y_std,
        y_min,
        y_max,
        units,
        n_hidden,
        n_output
    )

def summary(samples: PyTree, axis:PyTree=None) -> Tuple[PyTree]:

    if axis is None:
        return (tree_map(jnp.mean, samples), tree_map(jnp.std, samples))

    return (
        tree_map(jnp.mean, samples, axis),
        tree_map(jnp.std, samples, axis)
    )

def _standardise(x, mu, sigma):
    return (x - mu) / sigma

def _inverse_standardise(x, mu, sigma):
    return x * sigma + mu
