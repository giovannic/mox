import flax.linen as nn
import jax.numpy as jnp

class PositionalEncoding(nn.Module):
    d_model: int         # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim]
        # representing the positional encoding for max_len inputs
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:,None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (
            -jnp.log(10000.0) / self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x, t):
        x = x + self.pe[t]
        return x

class FillEncoding(nn.Module):

    @nn.compact
    def __call__(self, x, t, max_t):
        d = jnp.diff(jnp.concatenate([t, max_t.reshape(1,)]))
        return jnp.repeat(x, d, axis=0)
