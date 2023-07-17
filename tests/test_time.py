import jax.numpy as jnp
from jax import random
from flax.linen.module import _freeze_attr
from jax.tree_util import tree_structure
from mox.timeseries_embedding import (
    positional_encoding,
    fill_encoding
)
from mox.seq2seq_surrogates import (
    make_rnn_seq2seq_surrogate,
    make_transformer_seq2seq_surrogate,
    RecoverSeqStructure
)
from utils import assert_tree_equal

#TODO:
# 1. test positional encoding
# 2. test fill encoding
# 3. test rnn seq2seq surrogate specification
# 3.1 test output encoding and recovery for seq2seq surrogate
# 3.2 (optional) test transformer seq2seq surrogate specification
# 4. test training loop

def positional_encoder_works_for_2d_time_series():
    x_samples = [
        jnp.array([[1., 2., 3., 4.]])
    ]
    t = jnp.array([1., 2., 3., 4.])

    x_vec = positional_encoding(x_samples, t)
    
    expected_x_vec = jnp.array([1., 2., 3., 4.])
    assert jnp.array_equal(x_vec, expected_x_vec)

def timeseries_fill_works():
    max_timestep = 10
    x_seq = {
        'beta': fill_encoding(
            jnp.array([1, 2, 3, 4, 5]),
            jnp.range(5) * 2,
            max_timestep
        ),
        'ages': fill_encoding(
            jnp.array([[1, 2, 3]]),
            0,
            max_timestep
        )
    }
    x_seq_expected = {
        'beta': jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        'ages': jnp.repeat([[1, 2, 3]], 10, axis=1)
    }
    assert_tree_equal(x_seq, x_seq_expected)


def test_output_recovery_works_for_list_of_timeseries():
    y_vec = jnp.array([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ])
    y_expected = _freeze_attr([
        jnp.array([1., 2., 3.]),
        jnp.array([4., 5., 6.]),
        jnp.array([7., 8., 9.])
    ])
    y_shapes = [(1,), (2,)]
    rec = RecoverSeqStructure(y_shapes, tree_structure(y_expected), (1, 3))

    key = random.PRNGKey(42)
    params = rec.init(key, y_vec)
    y = rec.apply(params, y_vec)

    assert_tree_equal(y, y_expected)

def test_e2e_timeseries():
    x_static = {
        'gamma': jnp.array(1),
        'inf': jnp.array(.3)
    }
    x_seq = {
        'beta': jnp.range(10),
        'ages': jnp.repeat([[1, 2, 3]], 10, axis=1)
    }

    y_seq = jnp.range(25).reshape((5,5))

    model = make_rnn_seq2seq_surrogate(x_static, x_seq, y_seq)
    key = random.PRNGKey(42)
    params = model.init(key, x_static, x_seq, y_seq)
    assert params is not None
