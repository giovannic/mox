
from typing import List, Tuple, Any
from flax import linen as nn
from jax.tree_util import tree_unflatten
from jax import numpy as jnp

class RecoverSeq(nn.Module):
    """Recover. Recover output PyTree from vectorised neural net output"""

    y_shapes: List[Tuple]
    y_def: Any
    y_boundaries: Tuple

    def __call__(self, y):
        y_leaves = [
                leaf.reshape(y.shape[:1] + shape)
            for leaf, shape in 
            zip(jnp.split(y, self.y_boundaries[:-1], axis=1), self.y_shapes)
        ]
        return tree_unflatten(self.y_def, y_leaves)
