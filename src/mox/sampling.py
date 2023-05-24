from typing import Union, Dict
from collections.abc import Iterable
from jaxtyping import PyTree
import jax

ParamRange = Union[
    tuple[float, float],
    Iterable['ParamRange'],
    Dict[str, 'ParamRange']
]

def sample_lhs(ranges: list[ParamRange], n: int, key) -> list[PyTree]:
    """sample_lhs.

    Sample function parameters from a Latin Hypercube and re-scale to a
    specified range.

    :param ranges: A tuple, or iterable or dict of tuples, specifying the lower
    and upper bounds for each parameter.
    :type ranges: list[ParamRange]
    :param n: The number of samples to make
    :type n: int
    :param key: The key to use to seed RNG
    :rtype: list[PyTree]
    """
    flat_ranges, treedef = jax.tree_util.tree_flatten(ranges)
    assert(
        (flat_ranges[1] > flat_ranges[0]).all(),
        'All lower bounds must be greater less than the upper bounds'
    )
    sampler = LatinHypercube(d=flat_ranges.shape[0], seed=key)
    samples = sampler.random(n)
    scaled_samples = samples * (
        flat_ranges[1] - flat_ranges[0]
    ) + flat_ranges[0] 
    return jax.tree_util.tree_unflatten(treedef, scaled_samples)
