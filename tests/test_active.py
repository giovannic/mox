from unittest.mock import Mock, patch
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from mox.active import sample_towards_utility
from mox.surrogates import make_surrogate, minrelu
from mox.loss import mse
from flax.linen.module import _freeze_attr

def test_sample_towards_utility_respects_min():
    x_samples = _freeze_attr({
        'param1': jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        'param2': jnp.array([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
    })
    y_samples = _freeze_attr({
        'output1': jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        'output2': jnp.array([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
    })

    model = make_surrogate(
        tree_map(lambda x: x[0], x_samples),
        tree_map(lambda x: x[0], y_samples),
    )
    key = random.PRNGKey(42)
    params = model.init(key, x_samples)

    mock_minrelu = Mock(wraps=minrelu)
    with patch('mox.active.minrelu', mock_minrelu):
        _ = sample_towards_utility(
            x_samples,
            model,
            params,
            lambda x, model, params: -mse(x, model.apply(params, x)),
            x_min = tree_map(lambda _: 0., x_samples),
            epochs = 1
        )

    assert mock_minrelu.call_count == 2
