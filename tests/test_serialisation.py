from jax import numpy as jnp, random
from mox.surrogates import make_surrogate
from mox.seq2seq.rnn import DecoderLSTMCell, make_rnn_surrogate
from mox.surrogates import MLP
from flax import linen as nn
from .helpers.utils import assert_tree_equal
from flax.training import orbax_utils
import orbax.checkpoint 
import dataclasses

def test_surrogate_can_be_serialised_as_pytree(tmp_path):
    x = [{
        'param1': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'param2': {
            'subparam1': jnp.array([5.0, 6.0, 7.0, 8.0]),
            'subparam2': jnp.array([8.0, 9.0, 10.0, 11.0])
        }
    }]
    y = [jnp.array([1.0, -1.0, 3.0, -3.0])]
    model = make_surrogate(x, y)
    empty_model = make_surrogate(
        [{
            'param1': jnp.array([0.0]),
            'param2': {
                'subparam1': jnp.array([0.0]),
                'subparam2': jnp.array([0.0])
            }
        }],
        [jnp.array([0.0])]
    )

    ckpt = dataclasses.asdict(model)
    empty_ckpt = dataclasses.asdict(empty_model)

    path = tmp_path / 'checkpoint'
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(path, ckpt)
    loaded_ckpt = orbax_checkpointer.restore(path, item=empty_ckpt)

    assert_tree_equal(ckpt, loaded_ckpt)

def test_rnn_surrogate_can_be_serialised_as_pytree(tmp_path):
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
    model = make_rnn_surrogate(x, x_seq, x_t, n_steps, y)
    empty_x = [{
        'gamma': jnp.array([1]),
        'inf': jnp.array([3])
    }]

    empty_x_seq = [{
        'beta': jnp.arange(5).reshape(1, 5, 1),
        'ages': jnp.repeat(jnp.arange(3).reshape((1, 1, 3)), 5, axis=1),
    }]

    empty_x_t = jnp.arange(5) * 2

    empty_y = [jnp.arange(40).reshape((1, 8, 5))]

    empty_model = make_rnn_surrogate(
        empty_x,
        empty_x_seq,
        empty_x_t,
        n_steps,
        empty_y
    )
    ckpt = dataclasses.asdict(model)
    empty_ckpt = dataclasses.asdict(empty_model)

    path = tmp_path / 'checkpoint'
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(path, ckpt)
    loaded_ckpt = orbax_checkpointer.restore(path, item=empty_ckpt)

    assert_tree_equal(ckpt, loaded_ckpt)

def test_mlp_can_be_serialised_with_orbax(tmp_path):
    path = tmp_path / 'checkpoint'
    model = MLP(2, 2, 2, .1, True)
    params = model.init(random.PRNGKey(0), jnp.ones((1, 2)), False)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {'net': dataclasses.asdict(model), 'params': params}
    empty_checkpoint = {
        'net': dataclasses.asdict(model),
        'params': model.init(random.PRNGKey(42), jnp.ones((1, 2)), False)
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)
    loaded_ckpt = orbax_checkpointer.restore(path, item=empty_checkpoint)
    assert_tree_equal(ckpt, loaded_ckpt)

def test_rnn_can_be_serialised_with_orbax(tmp_path):
    path = tmp_path / 'checkpoint'
    model = nn.RNN(DecoderLSTMCell(2, 2))
    params = model.init(random.PRNGKey(0), jnp.ones((1, 2, 2)))
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt = {'cell': dataclasses.asdict(model.cell), 'params': params}
    empty_model = nn.RNN(DecoderLSTMCell(2, 2))
    empty_checkpoint = {
        'cell': dataclasses.asdict(empty_model.cell),
        'params': model.init(random.PRNGKey(42), jnp.ones((1, 2, 2)))
    }
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(path, ckpt, save_args=save_args)
    loaded_ckpt = orbax_checkpointer.restore(path, item=empty_checkpoint)
    assert_tree_equal(ckpt, loaded_ckpt)
