import jax.numpy as jnp
from jax import random
from flax.linen.module import _freeze_attr
from jax.tree_util import tree_structure, tree_leaves
from mox.seq2seq.encoding import (
    PositionalEncoding,
    FillEncoding
)
from mox.seq2seq.surrogates import RecoverSeq
from mox.seq2seq.rnn import make_rnn_surrogate, SequenceVectoriser
from mox.seq2seq.transformer import TransformerSurrogate
from utils import assert_tree_equal

#TODO:
# 3. test rnn encoder decoder seq2seq surrogate specification
# 3. test rnn seq2seq surrogate specification
# 3.1 test output encoding and recovery for seq2seq surrogate
# 3.2 (optional) test transformer seq2seq surrogate specification
# 4. test training loop

#TODO really though:
# recreate what you already have before you do all this crazy stuff
# 1. direct seq2seq with variable sequences
# 2. does seq_lengths work with batch same size?

def test_positional_encoder_works_for_5d_time_series():
    x = jnp.arange(30).reshape((5, 6))
    t = jnp.arange(5) * 2

    encoder = PositionalEncoding(6, 50)
    x_enc = encoder.apply({}, x, t)

    assert x_enc.shape == (5, 6)

def test_timeseries_fill_works():
    max_timestep = jnp.array(10)
    x = jnp.arange(5).reshape((5, 1))
    t = jnp.arange(5) * 2
    encoder = FillEncoding()
    x_enc = encoder.apply({}, x, t, max_timestep)
    x_enc_expected = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape((10, 1))
    assert jnp.array_equal(x_enc, x_enc_expected)

def test_sequence_vectoriser_works():
    x = [
        jnp.arange(5).reshape((1, 5, 1)),
        jnp.repeat(jnp.arange(9).reshape((1, 1, 3, 3)), 5, axis=1),
    ]
    x_vec_expected = jnp.array([[
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [4, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]])

    vectoriser = SequenceVectoriser(6, -1)
    x_vec = vectoriser.apply({}, x)
    assert jnp.array_equal(x_vec, x_vec_expected)

def test_output_recovery_works_for_list_of_timeseries():
    y_expected = [
        jnp.arange(3),
        jnp.arange(9).reshape((3, 3))
    ]
    y_vec = jnp.array([
        [0, 0, 1, 2],
        [1, 3, 4, 5],
        [2, 6, 7, 8]
    ])

    y_shapes = [leaf.shape[1:] for leaf in tree_leaves(y_expected)]
    y_boundaries = tuple([
        int(i) for i in
        jnp.cumsum(jnp.array([jnp.prod(jnp.array(s)) for s in y_shapes]))
    ])

    rec = RecoverSeq(
        y_shapes,
        tree_structure(y_expected),
        y_boundaries
    )

    y = rec.apply({}, y_vec)

    assert_tree_equal(y, y_expected)

def test_e2e_timeseries():
    x = _freeze_attr([{
        'gamma': jnp.array([1, 2]),
        'inf': jnp.array([3, 4])
    }])

    x_seq = _freeze_attr([{
        'beta': jnp.arange(10).reshape(2, 5, 1),
        'ages': jnp.repeat(jnp.arange(6).reshape((2, 1, 3)), 5, axis=1),
    }])

    x_t = jnp.arange(5) * 2

    y = _freeze_attr([jnp.arange(50).reshape((2, 5, 5))])

    model = make_rnn_surrogate(x, x_seq, x_t, y, max_t=jnp.array(10))
    key = random.PRNGKey(42)
    params = model.init(key, x, x_seq, x_t)
    y_hat = model.apply(params, x, x_seq, x_t)
    assert params is not None
    assert y_hat[0].shape == (2, 10, 5)