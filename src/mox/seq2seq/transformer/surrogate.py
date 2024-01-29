from jaxtyping import PyTree

from flax import linen as nn
from jax import vmap
from ...utils import (
    tree_leading_axes as tla
)

from ..rnn import RNNSurrogate

def init_surrogate(key, surrogate: RNNSurrogate, net: nn.Module, x):
    x_vec = vmap(surrogate.vectorise, in_axes=[tla(x)])(x)
    return net.init(key, x_vec, train=False)

def apply_surrogate(surrogate: RNNSurrogate, net: nn.Module, params: PyTree, x):
    x_vec = vmap(surrogate.vectorise, in_axes=[tla(x)])(x)
    y = net.apply(params, x_vec, train=False)
    return vmap(surrogate.recover)(y)

