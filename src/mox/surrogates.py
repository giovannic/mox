from typing import Callable, Any, Sequence 
from jaxtyping import Array, PyTree
from flax import linen as nn
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad, random
from jax.tree_util import tree_map, tree_structure, tree_flatten, tree_unflatten

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
        x = nn.Dense(self.n_output)(x)

class Vectoriser(nn.Module):

    x_mean: Sequence[PyTree]
    x_std: Sequence[PyTree]

    @nn.compact
    def __call__(self, x):
        x = self.standardise(x)
        x = tree_map(lambda x: x.reshape((x.shape[0], -1)), x)
        x, _ = tree_flatten(x)
        x = jnp.concatenate(x, axis=1)
        return x

    def standardise(self, x):
        return tree_map(
            lambda x, mu, sigma: (x - mu) / sigma,
            x,
            self.x_mean,
            self.x_std
        )

class Recover(nn.Module):

    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree

    @nn.compact
    def __call__(self, y):
        y = tree_unflatten(y, tree_structure(self.y_mean))
        return self.recover(y)

    def recover(self, y):
        return tree_map(
            lambda y, mu, sigma: y * sigma + mu,
            y,
            self.y_mean,
            self.y_std
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
    y_mean: PyTree
    y_std: PyTree
    y_min: PyTree
    y_max: PyTree
    units: int
    n_hidden: int
    n_output: int

    def setup(self):
        self.vec = Vectoriser(self.x_mean, self.x_std)
        self.rec = Recover(self.y_mean, self.y_std)
        self.nn = MLP(self.units, self.n_hidden, self.n_output)

    def __call__(self, x):
        x = self.vec(x)
        y = self.nn(x)
        return self.rec(y)

def make_surrogate(
        x: list[PyTree],
        y: PyTree,
        y_min: Array,
        y_max: Array,
        idx_max: Array,
        loss: Callable[[PyTree, Array, Array], float],
        key: Any
    ) -> tuple[nn.Module, PyTree]:
    """make_surrogate.

    Train a maskedminmax MLP surrogate and return a tuple including the flax
    module and parameters

    :param x: Function parameter samples
    :type x: list[PyTree]
    :param y: Function outputs
    :type y: Array
    :param y_min: Minimum function outputs
    :type y_min: Array
    :param y_max: Maximum function outputs
    :type y_max: Array
    :param idx_max: Mask for the maximum outputs
    :type idx_max: Array
    :param loss: Loss function for training
    :type loss: Callable[[PyTree, Array, Array], float]
    :rtype: tuple[nn.Module, PyTree]
    """
    y_shape = y.shape[1:]
    surrogate_model = MaskedMinMaxMLP(
        units=288,
        n_hidden=3,
        n_output=jnp.product(jnp.array(y_shape)),
        y_min=standardise(jnp.zeros(y_shape), y_mean, y_std)[0].reshape(-1),
        y_max=standardise(jnp.ones(y_shape), y_mean, y_std)[0].reshape(-1),
        idx_max=idx_max
    )
    surrogate_params = surrogate_model.init(key, x)

    tx = optax.adam(learning_rate=.001)
    opt_state = tx.init(surrogate_params)
    loss_grad_fn = value_and_grad(jit(loss))

    batch_size = 100

    n_batches = X.shape[0] // batch_size
    X_batched = jnp.reshape(X, (n_batches, batch_size, -1))
    y_batched = jnp.reshape(y, (n_batches, batch_size, -1))

    epochs = 100

    for i in range(epochs):
        key, key_i = random.split(key)

        for b in random.permutation(key_i, n_batches, independent=True):
            loss_val, grads = loss_grad_fn(
                surrogate_params,
                X_batched[b],
                y_batched[b]
            )
            updates, opt_state = tx.update(grads, opt_state)
            surrogate_params = optax.apply_updates(surrogate_params, updates)

    return (surrogate_model, surrogate_params)

def summary(samples, axis=None):

    if axis is None:
        return (tree_map(jnp.mean, samples), tree_map(jnp.std, samples))

    return (
        tree_map(jnp.mean, samples, axis),
        tree_map(jnp.std, samples, axis)
    )
