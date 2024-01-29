import jax.numpy as jnp
from jax import random
from mox.seq2seq.rnn import make_rnn_surrogate
from mox.seq2seq.transformer.surrogate import (
    init_surrogate,
    apply_surrogate
)

from mox.seq2seq.transformer.transformer import DensityTransformer
from mox.seq2seq.transformer.training import train_transformer
from jax.scipy.stats import norm

def test_transformer_training():
    x = [{
        'gamma': jnp.array([1, 2]),
        'inf': jnp.array([3, 4])
    }]

    x_seq = [{
        'beta': jnp.arange(10).reshape(2, 5, 1),
        'ages': jnp.repeat(jnp.arange(6).reshape((2, 1, 3)), 5, axis=1),
    }]

    x_t = jnp.arange(5) * 2

    y = [jnp.arange(80).reshape((2, 8, 5))]

    n_steps = jnp.max(x_t)
    model = make_rnn_surrogate(x, x_seq, x_t, n_steps, y, density=True)
    key = random.PRNGKey(42)
    x_in = (x, x_seq)

    net = DensityTransformer(
        num_layers = 1,
        latent_dim = 16,
        output_dim = model.get_output_dim(y),
        num_heads = 2,
        dim_feedforward = 32,
        dropout_prob = 0.1
    )
    params = init_surrogate(key, model, net, x_in)
    mu, log_std = apply_surrogate(model, net, params, x_in)
    assert params is not None
    assert mu[0].shape == (2, 8, 5)
    assert log_std[0].shape == (2, 8, 5)

    # Define loss function
    def loss(y_hat, y):
        mu, logsigma = y_hat
        sigma = jnp.exp(logsigma)
        return -jnp.sum(norm.logpdf(
            y,
            loc=mu,
            scale=sigma
        ))

    params = train_transformer(
        x_in,
        y,
        model,
        net,
        params,
        loss,
        key,
        epochs = 1,
        batch_size = 1
    )
    assert params is not None
