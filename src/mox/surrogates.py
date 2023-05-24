from typing import Callable
from jaxtyping import Array

def maskedminmaxrelu(x: Array, min_x: Array, max_x: Array, idx: Array) -> Array:
    """maskedminmaxrelu.

    relu with a min and max input, however max_x is only set at idx

    :param x: input array
    :type x: Array
    :param min_x: minimum value
    :type min_x: Array
    :param max_x: maximum value
    :type max_x: Array
    :param idx: mask for the maximum
    :type idx: Array
    :rtype: Array
    """
    filtered_min = jnp.maximum(x, min_x)
    filtered_max = jnp.minimum(filtered_min, max_x)
    return filtered_min.at[idx].set(filtered_max[idx])

class MaskedMinMaxMLP(nn.Module):
    """MaskedMinMaxMLP.

    MLP model with a maskedminmaxrelu as an output
    """

    units: int
    n_hidden: int
    n_output: int
    y_min: Array
    y_max: Array
    idx_max: Array

    @nn.compact
    def __call__(self, x):
        layers = [nn.Dense(self.units) for _ in range(self.n_hidden)]
        for i, lyr in enumerate(layers):
            x = lyr(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_output)(x)
        return maskedminmaxrelu(x, self.y_min, self.y_max, self.idx_max)

class StandardisedSurrogate():
    """StandardisedSurrogate.

    A wrapper class to standardise surrogate inputs and unstandardise surrogate
    outputs.
    """

    x_mean: PyTree
    x_std: PyTree 
    y_mean: Array
    y_std: Array
    network: nn.Module
    params: PyTree

    def __call__(self, x):
        return unstandardise(
            network.apply(params, standardise(x, x_mean, x_std)),
            y_mean,
            y_std
        )

def make_mmm_surrogate(
        x: list[PyTree],
        y: Array,
        y_min: Array,
        y_max: Array,
        idx_max: Array,
        loss: Callable[[PyTree, Array, Array], float]
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
