from typing import Callable, Any, Optional
from jaxtyping import Array
import flax.linen as nn
from jax import numpy as jnp, lax

class PositionalEmbedding(nn.Module):
    """Positional Embedding.

    Attributes:
    d_feature: latent dimension.
    decode: whether to use autoregressive decoding.
    """

    d_feature: int
    dtype: Any

    def setup(self):
        self.pe = _position(self.d_feature, dtype=self.dtype)

    def __call__(self, inputs): # type: ignore
        x = inputs + self.pe[:, :inputs.shape[1], :]
        return x

class MLP(nn.Module):
    """Transformer MLP block.

    Attributes:
    latent_dim: latent dimension.
    out_dim: output dimension.
    dropout: dropout rate.
    activation: activation function.
    """

    latent_dim: int
    out_dim: int
    dropout: float
    activation: Callable

    @nn.compact
    def __call__(self, inputs, deterministic=True): # type: ignore
        x = nn.Dense(self.latent_dim)(inputs)
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        output = nn.Dense(self.out_dim)(x)
        output = nn.Dropout(rate=self.dropout)(
            output, deterministic=deterministic
        )
        return output

class EncoderBlock(nn.Module):
    """Transformer Encoder Block.

    Attributes:
    input_dim: input dimension.
    num_heads: number of heads.
    latent_dim: latent dimension.
    dim_feedforward: feedforward dimension.
    dropout: dropout rate.
    """

    num_heads : int
    latent_dim : int
    dim_feedforward : int
    dropout : float

    @nn.compact
    def __call__ (# type: ignore
        self,
        x: Array,
        deterministic: bool,
        encoder_mask: Optional[Array]=None
    ):
        x_att = nn.LayerNorm()(x)
        x_att = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.latent_dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x_att, mask=encoder_mask)
        x_att = nn.Dropout(rate=self.dropout)(
            x_att,
            deterministic=deterministic
        )
        x = x_att + x

        y = nn.LayerNorm()(x)
        y = MLP(
            latent_dim=self.dim_feedforward,
            out_dim=self.latent_dim,
            dropout=self.dropout,
            activation=nn.relu
        )(y, deterministic=deterministic)
        return x + y

class Encoder(nn.Module):
    """Transformer Encoder.

    Attributes:
    num_layers: number of layers.
    output_dim: output dimension.
    num_heads: number of heads.
    dim_feedforward: feedforward dimension.
    dropout_prob: dropout rate.
    """

    num_layers : int
    latent_dim : int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    @nn.compact
    def __call__(self, x, train=True, encoder_mask=None): # type: ignore
        x = PositionalEmbedding(
            self.latent_dim,
            lax.dtype(x.dtype)
        )(x)
        for _ in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads,
                latent_dim=self.latent_dim,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_prob,
            )(x, deterministic=not train, encoder_mask=encoder_mask)

        x = nn.LayerNorm()(x)
        return x

class DensityTransformer(nn.Module):
    """Density Transformer.

    Attributes:
    num_layers: number of layers.
    latent_dim: latent dimension.
    num_heads: number of heads.
    dim_feedforward: feedforward dimension.
    dropout_prob: dropout rate.
    padding_value: padding value.
    """

    num_layers : int
    latent_dim : int
    output_dim: int
    num_heads : int
    dim_feedforward : int
    dropout_prob : float

    @nn.compact
    def __call__( # type: ignore
        self,
        x: Array,
        train: bool=True
        ):
        x = nn.Dense(self.latent_dim)(x)
        mask = nn.make_causal_mask(to_1d(x))
        x = Encoder(
            num_layers=self.num_layers,
            latent_dim=self.latent_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout_prob,
        )(x, train=train, encoder_mask=mask)
        mu = nn.Dense(self.output_dim)(x)
        log_std = nn.Dense(self.output_dim)(x)
        return mu, log_std

def to_1d(x):
    return x[..., 0]

def _position(d_feature, dtype, max_len = 1024):
    pe = jnp.zeros((max_len, d_feature), dtype=dtype)
    position = jnp.arange(0, max_len, dtype=dtype)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, d_feature, 2) * -(jnp.log(10000.0) / d_feature)
    )
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe[jnp.newaxis, :, :]