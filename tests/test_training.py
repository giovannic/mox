from mox.training import train_surrogate
from mox.surrogates import make_surrogate, init_surrogate, MLP
from mox.loss import mse
from jax import numpy as jnp, random

def test_training_with_batch_norm_and_dropout():
    x = [{
        'param1': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'param2': {
            'subparam1': jnp.array([5.0, 6.0, 7.0, 8.0]),
            'subparam2': jnp.array([8.0, 9.0, 10.0, 11.0])
        }
    }]
    y = [jnp.array([1.0, -1.0, 3.0, -3.0])]
    key = random.PRNGKey(42)
    model = make_surrogate(x, y)
    net = MLP(2, 2, 1, dropout_rate=0.5, batch_norm=True)
    variables = init_surrogate(key, model, net, x)
    state = train_surrogate(
        x,
        y,
        model,
        net,
        mse,
        key,
        variables,
        batch_size=2,
        epochs=1
    )
    assert state.batch_stats is not None

def test_training_wo_batch_norm():
    x = [{
        'param1': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'param2': {
            'subparam1': jnp.array([5.0, 6.0, 7.0, 8.0]),
            'subparam2': jnp.array([8.0, 9.0, 10.0, 11.0])
        }
    }]
    y = [jnp.array([1.0, -1.0, 3.0, -3.0])]
    key = random.PRNGKey(42)
    model = make_surrogate(
        x,
        y
    )
    net = MLP(2, 2, 1, dropout_rate=0.5, batch_norm=True)
    variables = init_surrogate(key, model, net, x)
    state = train_surrogate(
        x,
        y,
        model,
        net,
        mse,
        key,
        variables,
        batch_size=2,
        epochs=1
    )
    assert state.params is not None
