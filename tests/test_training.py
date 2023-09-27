from mox.training import batch_tree
from jax import numpy as jnp
from utils import assert_tree_equal
from jax import jit
from mox.loss import l2_loss

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

def test_l2_loss_calculates_penalty_correctly():
    params = {
        'params': {
            'nn': {
                'Dense_0': {
                    'bias': jnp.full((10,), 3.),
                    'kernel': jnp.full((10,), 2.)
                }
            }
        }
    }

    loss = jit(l2_loss)(params, 0.01)
    assert loss == 0.01 * 2.**2 
