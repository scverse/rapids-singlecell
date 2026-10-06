from __future__ import annotations

from collections.abc import Sequence
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.random import check_random_state

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from numpy.random import BitGenerator

__all__ = [
    "RNGLike",
    "SeedLike",
    "_LegacyRandom",
    "_LegacyRng",
    "_accepts_legacy_random_state",
    "_if_legacy_apply_global",
    "_legacy_random_state",
    "_seed_from_rng",
]

type SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
type RNGLike = np.random.Generator | np.random.BitGenerator
type _LegacyRandom = int | np.random.RandomState | None

_SEED_BOUND = 2**32
"""cuML, cuGraph and CuPy all take a 32-bit unsigned seed."""


###################################
# Compatibility with legacy numpy #
###################################


class _LegacyRng(np.random.Generator):
    """A `Generator` that wraps a legacy `RandomState` instance.

    To behave like a `RandomState`, it's not enough to just use a MT19937
    `bit_generator` (as in `Generator(RandomState(seed).bit_generator)`),
    so instead this hack uses the exact same random numbers as `RandomState(seed)`.
    """

    arg: _LegacyRandom
    state: np.random.RandomState

    def __init__(
        self, arg: _LegacyRandom, state: np.random.RandomState | None = None
    ) -> None:
        self.arg = arg
        self.state = check_random_state(arg) if state is None else state

    def __str__(self) -> str:
        return f"LegacyRng({self.arg!r})"

    @property
    def bit_generator(self) -> BitGenerator:
        msg = "A _LegacyRng instance has no `bit_generator` attribute."
        raise AttributeError(msg)

    @classmethod
    def wrap_global(
        cls, arg: _LegacyRandom = None, state: np.random.RandomState | None = None
    ) -> Self:
        """Create a generator that wraps the global `RandomState` backing the legacy `np.random` functions."""
        if arg is not None:
            if isinstance(arg, np.random.RandomState):
                np.random.set_state(arg.get_state(legacy=False))
                return cls(arg, state)
            np.random.seed(arg)
        return cls(arg, np.random.RandomState(np.random.get_bit_generator()))

    def spawn(self, n_children: int) -> list[Self]:
        """Return `self` `n_children` times.

        In a real generator, the spawned children are independent, but for
        backwards compatibility we return the same instance so that its internal
        state is advanced by each child.
        """
        return [self] * n_children

    @classmethod
    def _delegate(cls) -> None:
        names = {"integers": "randint"}
        for name, meth in np.random.Generator.__dict__.items():
            if name.startswith("_") or not callable(meth) or name in cls.__dict__:
                continue

            def mk_wrapper(name: str, meth):
                @wraps(meth, assigned=set(WRAPPER_ASSIGNMENTS) - {"__doc__"})
                def wrapper(self: _LegacyRng, *args, **kwargs):
                    return getattr(self.state, name)(*args, **kwargs)

                return wrapper

            setattr(cls, name, mk_wrapper(names.get(name, name), meth))


_LegacyRng._delegate()


def _if_legacy_apply_global(rng: np.random.Generator, /) -> np.random.Generator:
    """Wrap the global legacy RNG if `rng` is a `_LegacyRng`.

    This is used where our code used to call `np.random.seed()`.
    It's a no-op if `rng` is not a `_LegacyRng`.
    """
    if not isinstance(rng, _LegacyRng):
        return rng
    return _LegacyRng.wrap_global(rng.arg, rng.state)


def _legacy_random_state(
    rng: SeedLike | RNGLike | None, /, *, always_state: bool = False
) -> _LegacyRandom:
    """Convert a np.random.Generator into a legacy `random_state` argument.

    If `rng` is already a `_LegacyRng`, return its original `arg` attribute.
    """
    if isinstance(rng, _LegacyRng):
        return rng.state if always_state else rng.arg
    [bitgen] = np.random.default_rng(rng).bit_generator.spawn(1)
    return np.random.RandomState(bitgen)


def _accepts_legacy_random_state[**P, R](
    random_state_default: _LegacyRandom, /
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Make a function accept `random_state: _LegacyRandom` and pass it as `rng`.

    If the decorated function is called with a `random_state` argument,
    it'll be wrapped in a `_LegacyRng`.
    Passing both `rng` and `random_state` at the same time is an error.
    If neither is given, `random_state_default` is used.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            match "random_state" in kwargs, "rng" in kwargs:
                case True, True:
                    msg = "Specify at most one of `rng` and `random_state`."
                    raise TypeError(msg)
                case True, False:
                    kwargs["rng"] = _LegacyRng(kwargs.pop("random_state"))
                case False, False:
                    kwargs["rng"] = _LegacyRng(random_state_default)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _seed_from_rng(
    rng: SeedLike | RNGLike | None, /, *, allow_none: bool = True
) -> int | None:
    """Draw the integer seed a cuML, cuGraph or CuPy call needs.

    Scanpy inlines this where it calls into `leidenalg`; we need it in more
    places, since most of what we wrap is seeded rather than state-based.
    Call it *at* such a call, so that everything upstream keeps passing the
    generator around and two consumers get two independent seeds.

    A legacy `random_state` integer is forwarded untouched so that existing
    calls keep producing the same results.

    `random_state=None` means unseeded. cuML and cuGraph interpret `None` that
    way themselves, so it is passed through; pass `allow_none=False` where a
    concrete integer is required instead, e.g. a CUDA kernel or `cupy.random`.
    """
    if isinstance(rng, _LegacyRng):
        if rng.arg is None:
            return None if allow_none else int(rng.integers(0, _SEED_BOUND))
        if isinstance(rng.arg, (int, np.integer)):
            return int(rng.arg)
        return int(rng.arg.randint(0, _SEED_BOUND))
    return int(np.random.default_rng(rng).integers(0, _SEED_BOUND))
