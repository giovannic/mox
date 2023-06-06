from mox.training import batch_tree, nn_loss
from mox.surrogates import make_surrogate
from jax import numpy as jnp
from utils import assert_tree_equal
from jax import random
from unittest.mock import MagicMock
from flax.linen.module import _freeze_attr

def test_batch_tree_with_divisible_samples():
    samples = {
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        'param2': jnp.array([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
    }
    batch_size = 2

    batched_samples = batch_tree(samples, batch_size)

    assert len(batched_samples) == 2
    assert_tree_equal(batched_samples[0], {
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'param2': jnp.array([[9.0, 10.0], [11.0, 12.0]])
    })
    assert_tree_equal(batched_samples[1], {
        'param1': jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        'param2': jnp.array([[13.0, 14.0], [15.0, 16.0]])
    })

def test_nn_loss_extracts_vectorised_outputs():
    x_samples = _freeze_attr([{
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'param2': jnp.array([[5.0, 6.0], [7.0, 8.0]])
    }])
    y_samples = {
        'output1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        'output2': jnp.array([[5.0, 6.0], [7.0, 8.0]])
    }

    model = make_surrogate(x_samples, y_samples)
    key = random.PRNGKey(42)
    params = model.init(key, x_samples)

    loss = MagicMock()
    _ = nn_loss(model, params, loss, x_samples, y_samples)
    loss.assert_called_once()
    assert loss.call_args_list[0][0][0].shape == (2, 4)
    assert loss.call_args_list[0][0][1].shape == (2, 4)
