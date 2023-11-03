import jax.numpy as jnp
from jax import random, vmap
from flax.linen.module import _freeze_attr
from jax.tree_util import tree_structure, tree_leaves, tree_map
from mox.seq2seq.rnn import (
    make_rnn_surrogate,
    _fill,
    _filler,
    _vectorise_sequence,
    _recover_sequence,
    init_surrogate,
    apply_surrogate
)
from mox.seq2seq.training import train_rnn_surrogate
from mox.utils import tree_leading_axes as tla
from .helpers.utils import assert_tree_equal
from mox.loss import mse

def test_timeseries_fill_works():
    max_timestep = jnp.array(10)
    x = jnp.arange(5).reshape((5, 1))
    t = jnp.arange(5) * 2
    x_enc = _fill(x, _filler(t, max_timestep), max_timestep)
    x_enc_expected = jnp.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]).reshape((10, 1))
    assert jnp.array_equal(x_enc, x_enc_expected)

def test_sequence_vectoriser_works():
    x = [
        jnp.arange(5).reshape((5, 1)),
        jnp.repeat(jnp.arange(9).reshape((1, 3, 3)), 5, axis=0),
    ]
    x_vec_expected = jnp.array([
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [4, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ])

    x_vec = _vectorise_sequence(x)
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
    y_def = tree_structure(y_expected)

    y = _recover_sequence(y_vec, y_shapes, y_def, y_boundaries, jnp.array(3))

    assert_tree_equal(y, y_expected)

def test_surrogate_can_vectorise_an_input_pair():
    x = _freeze_attr([{
        'gamma': jnp.array([1, 2]),
        'inf': jnp.array([3, 4])
    }])

    x_seq = _freeze_attr([{
        'beta': jnp.array([[[1.], [2.]], [[1.], [2.]]]),
        'ages': jnp.array([[[0.], [2.]], [[0.], [2.]]]),
    }])

    x_t = jnp.arange(2)

    y = _freeze_attr([jnp.arange(20).reshape((2, 2, 5))])

    n_steps = jnp.array(2)
    model = make_rnn_surrogate(
        x,
        x_seq,
        x_t,
        n_steps,
        y
    )
    x_in = (x, x_seq)
    x_vec = vmap(model.vectorise, in_axes=[tla(x_in)])(x_in)
    x_expected = jnp.array([
        [
            [-1., -1., -1., -1.],
            [-1., -1., 1., 1.]
        ],[
            [1., 1., -1., -1.],
            [1., 1., 1., 1.]
        ]
    ])
    assert jnp.array_equal(x_vec, x_expected)

def test_surrogate_vectorise_output_and_recover_are_consistent():
    x = _freeze_attr([{
        'gamma': jnp.array([1, 2], dtype=jnp.float64),
        'inf': jnp.array([3, 4], dtype=jnp.float64)
    }])

    x_seq = _freeze_attr([{
        'beta': jnp.array([[[0.], [2.]], [[1.], [3.]]], dtype=jnp.float64),
        'ages': jnp.array([[[0.], [2.]], [[1.], [3.]]], dtype=jnp.float64) + 1.,
    }])

    x_t = jnp.arange(2)

    y_expected = _freeze_attr([{
        'inc': jnp.array([[[0.], [2.]], [[1.], [3.]]], dtype=jnp.float64),
        'prev': jnp.array([[[0.], [2.]], [[1.], [3.]]], dtype=jnp.float64) + 1.
    }])

    n_steps = jnp.array(2)
    model = make_rnn_surrogate(
        x,
        x_seq,
        x_t,
        n_steps,
        y_expected,
        y_var_axis=tree_map(lambda _: (0, 1), y_expected)
    )
    y_vec = vmap(
        model.vectorise_output,
        in_axes=[tla(y_expected)]
    )(y_expected)
    assert y_vec.shape == (2, 2, 2)
    y = vmap(model.recover)(y_vec)
    assert_tree_equal(y, y_expected, 1e-5)

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

    n_steps = jnp.max(x_t)
    model = make_rnn_surrogate(x, x_seq, x_t, n_steps, y)
    key = random.PRNGKey(42)
    x_in = (x, x_seq)
    params = init_surrogate(key, model, x_in)
    y_hat = apply_surrogate(model, params, x_in)
    assert params is not None
    assert y_hat[0].shape == (2, 5, 5)
    params = train_rnn_surrogate(
        x_in,
        y,
        model,
        params,
        mse,
        key,
        epochs = 1,
        batch_size = 1
    )
    assert params is not None
