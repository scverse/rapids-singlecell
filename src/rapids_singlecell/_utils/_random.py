from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "RNGLike",
    "SeedLike",
    "_LegacyRandom",
    "_LegacyRng",
    "_accepts_legacy_random_state",
    "_legacy_random_state",
    "_seed_from_rng",
]

type SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
type RNGLike = np.random.Generator | np.random.BitGenerator
type _LegacyRandom = int | np.random.RandomState | None

_SEED_BOUND = 2**32
"""cuML, cuGraph and CuPy all take a 32-bit unsigned seed."""


class _LegacyRng:
    """Marks a seed that arrived through the superseded `random_state` argument.

    Unlike scanpy's class of the same name, this is only a marker. Integer-only
    GPU consumers use it to recover the legacy seed, while host-side consumers
    can recover the original :class:`~numpy.random.RandomState` object and
    continue its exact stream.
    """

    __slots__ = ("arg",)

    def __init__(self, arg: _LegacyRandom) -> None:
        self.arg = arg

    def __repr__(self) -> str:
        return f"_LegacyRng({self.arg!r})"


def _accepts_legacy_random_state[**P, R](
    default: _LegacyRandom, /
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Let a function taking `rng` still be called with `random_state`.

    A `random_state` argument is wrapped in a :class:`_LegacyRng` and passed as
    `rng`. Passing both is an error. If neither is given, `default` is used.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            match "random_state" in kwargs, "rng" in kwargs:
                case True, True:
                    raise TypeError("Specify at most one of `rng` and `random_state`.")
                case True, False:
                    kwargs["rng"] = _LegacyRng(kwargs.pop("random_state"))
                case False, False:
                    kwargs["rng"] = _LegacyRng(default)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _seed_from_rng(rng: SeedLike | RNGLike | _LegacyRng | None, /) -> int | None:
    """The integer seed to hand to cuML, cuGraph, CuPy or a kernel.

    A `random_state` integer is forwarded untouched so that existing calls keep
    producing the same results; anything else is drawn from the generator.
    """
    if isinstance(rng, _LegacyRng):
        if rng.arg is None:
            return None
        if isinstance(rng.arg, (int, np.integer)):
            return int(rng.arg)
        return int(rng.arg.randint(0, _SEED_BOUND))
    return int(np.random.default_rng(rng).integers(0, _SEED_BOUND))


def _legacy_random_state(
    rng: SeedLike | RNGLike | _LegacyRng | None, /
) -> _LegacyRandom:
    """The value to hand to host-side APIs that still take a `random_state`."""
    if isinstance(rng, _LegacyRng):
        return rng.arg
    [bit_generator] = np.random.default_rng(rng).bit_generator.spawn(1)
    return np.random.RandomState(bit_generator)
