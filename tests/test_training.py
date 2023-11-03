from mox.training import train_surrogate
from mox.surrogates import make_surrogate, init_surrogate
from mox.loss import mse
from jax import numpy as jnp, random
from flax.linen.module import _freeze_attr

def test_training_with_batch_norm_and_dropout():
    x = _freeze_attr([{
        'param1': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'param2': {
            'subparam1': jnp.array([5.0, 6.0, 7.0, 8.0]),
            'subparam2': jnp.array([8.0, 9.0, 10.0, 11.0])
        }
    }])
    y = [jnp.array([1.0, -1.0, 3.0, -3.0])]
    key = random.PRNGKey(42)
    model = make_surrogate(x, y)
    variables = init_surrogate(key, model, x)
    state = train_surrogate(
        x,
        y,
        model,
        mse,
        key,
        variables,
        batch_size=2,
        epochs=1
    )
    assert state.batch_stats is not None

def test_training_wo_batch_norm():
    x = _freeze_attr([{
        'param1': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'param2': {
            'subparam1': jnp.array([5.0, 6.0, 7.0, 8.0]),
            'subparam2': jnp.array([8.0, 9.0, 10.0, 11.0])
        }
    }])
    y = [jnp.array([1.0, -1.0, 3.0, -3.0])]
    key = random.PRNGKey(42)
    model = make_surrogate(
        x,
        y,
        batch_norm=False
    )
    variables = init_surrogate(key, model, x)
    state = train_surrogate(
        x,
        y,
        model,
        mse,
        key,
        variables,
        batch_size=2,
        epochs=1
    )
    assert state.params is not None
