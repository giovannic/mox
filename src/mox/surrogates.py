from typing import Sequence, List, Tuple, Any, Optional, Union
from jaxtyping import Array, PyTree
from flax import linen as nn
from flax.linen import Module
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

class MLP(Module):
    """MLP. A multi layer perceptron
    """

    units: int
    n_hidden: int
    n_output: Union[int, Array]

    @nn.compact
    def __call__(self, x):
        layers = [nn.Dense(self.units) for _ in range(self.n_hidden)]
        for i, lyr in enumerate(layers):
            x = lyr(x)
            x = nn.relu(x)
        return nn.Dense(self.n_output)(x)

class Vectoriser(Module):

    @nn.compact
    def __call__(self, x):
        return tree_to_vector(x)

class Standardiser(nn.Module):

    x_mean: Sequence[PyTree]
    x_std: Sequence[PyTree]

    @nn.compact
    def __call__(self, x):
        return tree_map(_standardise, x, self.x_mean, self.x_std)

class InverseStandardiser(nn.Module):

    x_mean: PyTree
    x_std: PyTree

    @nn.compact
    def __call__(self, x):
        return tree_map(_inverse_standardise, x, self.x_mean, self.x_std)

class Recover(nn.Module):
    """Recover. Recover output PyTree from vectorised neural net output"""

    y_shapes: List[Tuple]
    y_def: Any
    y_boundaries: Tuple

    def __call__(self, y):
        y_leaves = [
            leaf.reshape(shape)
            for leaf, shape in 
            zip(jnp.split(y, self.y_boundaries), self.y_shapes)
        ]
        return tree_unflatten(self.y_def, y_leaves)

def _take_leaf(vector: Array, start: int, end: int, shape: tuple):
    return vector[start:end].reshape(shape)

class Limiter(nn.Module):
    """Limiter. limit outputs using relus"""

    y_min: PyTree
    y_max: PyTree

    @nn.compact
    def __call__(self, y):
        return tree_map(
            lambda y, y_min, y_max: maxrelu(minrelu(y, y_min), y_max),
            y,
            self.y_min,
            self.y_max
        )

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
    y_shapes: List[Tuple]
    y_boundaries: Tuple
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_hidden: int
    n_output: Union[int, Array]

    def setup(self):
        self.vec = Vectoriser()
        self.std = Standardiser(self.x_mean, self.x_std)
        self.rec = Recover(
            self.y_shapes,
            tree_structure(self.y_mean),
            self.y_boundaries
        )
        self.inv_std = InverseStandardiser(self.y_mean, self.y_std)
        self.nn = MLP(self.units, self.n_hidden, self.n_output)
        if self.y_min is not None:
            self.limiter = Limiter(self.y_min, self.y_max)
        else:
            self.limiter = lambda x: x

    def __call__(self, x):
        x = self.std(x)
        x = self.vec(x)
        y = self.nn(x)
        y = self.rec(y)
        y = self.limiter(y)
        y = self.inv_std(y)
        return y

def make_surrogate(
        x: list[PyTree],
        y: PyTree,
        nn: Any=MLP,
        x_std_axis: Optional[PyTree] = None,
        y_std_axis: Optional[PyTree] = None,
        y_min: Optional[Array] = None,
        y_max: Optional[Array] = None,
        units = 256,
        n_hidden = 3
    ) -> nn.Module:
    """make_surrogate.

    Make a surrogate model from a function samples
    """
    x_mean, x_std = summary(x, x_std_axis)
    y_mean, y_std = summary(y, y_std_axis)
    y_shapes = [leaf.shape[1:] for leaf in tree_leaves(y)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])
    n_output = y_boundaries[-1]
    return Surrogate(
        x_mean,
        x_std,
        y_shapes,
        y_boundaries,
        y_mean,
        y_std,
        y_min,
        y_max,
        units,
        n_hidden,
        n_output
    )

def summary(samples: PyTree, axis:PyTree=None) -> Tuple[PyTree, PyTree]:

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
