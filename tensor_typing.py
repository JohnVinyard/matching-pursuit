"""
tensor_typing.py: nominal, "strongly-typed" tensor dimensions for numpy / jax / PyTorch.

Only uses the standard library. Arrays are treated as duck types: anything with a
``.shape`` and either ``.permute(*axes)`` (torch) or ``.transpose(axes)`` (numpy, jax).

Core idea
---------
Each dimension is a *nominal* type, declared by the user with ``NewType``::

    Batch = NewType("Batch", Dim)
    Channels = NewType("Channels", Dim)
    Time = NewType("Time", Dim)
    Frequency = NewType("Frequency", Dim)

    Audio = Tensor[Batch, Channels, Time]
    Spectrogram = Tensor[Batch, Frequency, Channels, Time]

``Tensor`` is generic over a TypeVarTuple, and TypeVarTuples are invariant, so a
static checker (pyright / mypy) treats ``Tensor[Batch, Channels, Time]`` and
``Tensor[Batch, Time, Channels]`` as incompatible. Two different ``NewType``s are
also distinct, even if they wrap the same base, so ``Time`` is never mistaken for
``Frequency``.

``Identity`` is an alias for ``typing.Any``. Static checkers accept any dimension in
an ``Identity`` slot, and the runtime checks here do the same.

A ``Tensor`` is a thin wrapper holding ``.data``, the raw backend array, and ``.dims``,
the runtime tuple of dimension types. Because the dims exist at runtime, layout
utilities such as ``permute`` can compute axis orders from the types alone. You
name the type you want and the library works out how to get there.

What the static checker can and cannot do
-----------------------------------------
Python's type system cannot compute a permutation at the type level. Operations
that change layout therefore take the target type explicitly (``to=SomeType``).
The checker then knows the result type, and the runtime confirms that the target
is a valid rearrangement of the source. This is more verbose than einops-style
strings, but every layout in the program is a named, checkable type.
"""

from __future__ import annotations

import functools
import inspect
import operator
from collections import deque
from typing import (
    Any,
    Callable,
    Generic,
    NewType,
    ParamSpec,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

__all__ = [
    "Dim",
    "Identity",
    "Tensor",
    "DimError",
    "dims_of",
    "typed",
    "check",
    "permute",
    "relabel",
    "transform",
    "via",
    "via_raw",
    "checked",
]


# ---------------------------------------------------------------------------
# Dimension types
# ---------------------------------------------------------------------------


class Dim:
    """Base class for dimensions. Declare dimensions with ``NewType("Name", Dim)``."""


Identity: TypeAlias = Any
"""Wildcard dimension, compatible with any other dimension (statically and at runtime)."""


class DimError(TypeError):
    """Raised when a tensor's runtime dimensions do not fit the requested type."""


def _is_identity(d: Any) -> bool:
    return d is Any


def _dim_name(d: Any) -> str:
    if _is_identity(d):
        return "Identity"
    return getattr(d, "__name__", repr(d))


def _fmt(dims: tuple[Any, ...]) -> str:
    return "[" + ", ".join(_dim_name(d) for d in dims) + "]"


def _compatible(a: Any, b: Any) -> bool:
    return _is_identity(a) or _is_identity(b) or a is b


# ---------------------------------------------------------------------------
# Tensor wrapper
# ---------------------------------------------------------------------------

Ts = TypeVarTuple("Ts")


class Tensor(Generic[*Ts]):
    """
    A raw array (``.data``) together with its runtime dimension types (``.dims``).

    Don't construct directly; use ``typed(SomeTensorType, array)``.
    """

    __slots__ = ("data", "dims")

    def __init__(self, data: Any, dims: tuple[Any, ...]) -> None:
        self.data = data
        self.dims = dims

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.data)

    @property
    def ndim(self) -> int:
        return len(self.dims)

    def axis(self, dim: Any) -> int:
        """Integer axis of a named dimension (it must occur exactly once)."""
        hits = [i for i, d in enumerate(self.dims) if d is dim]
        if len(hits) != 1:
            raise DimError(
                f"dimension {_dim_name(dim)} occurs {len(hits)} times in {_fmt(self.dims)}"
            )
        return hits[0]

    def size(self, dim: Any) -> int:
        """Size of a named dimension."""
        return self.shape[self.axis(dim)]

    def sizes(self) -> dict[str, int]:
        return {_dim_name(d): s for d, s in zip(self.dims, self.shape)}

    def map(self, fn: Callable[[Any], Any]) -> Tensor[*Ts]:
        """Apply a layout-preserving function (elementwise ops, etc.) to the raw data."""
        out = fn(self.data)
        _check_rank(out, self.dims)
        return Tensor(out, self.dims)

    # --- Broadcasting arithmetic -------------------------------------------
    #
    # Statically, both operands must have the same dim types, but an Identity
    # (= Any) slot unifies with anything. So
    #     Tensor[Batch, Channels, Time] * Tensor[Identity, Channels, Time]
    # type-checks, while
    #     Tensor[Batch, Channels, Time] * Tensor[Batch, Frequency, Time]
    # does not. At runtime, each axis must have compatible dims and sizes that
    # broadcast (equal, or one of them is 1). The result keeps the more specific
    # dim on every axis.

    def _binary(self, other: Any, op: Callable[[Any, Any], Any], reflected: bool = False) -> Tensor[*Ts]:
        if not isinstance(other, Tensor):  # scalar
            out = op(other, self.data) if reflected else op(self.data, other)
            return Tensor(out, self.dims)
        dims = _broadcast_dims(self, other)
        out = op(other.data, self.data) if reflected else op(self.data, other.data)
        return Tensor(out, dims)

    def __add__(self, other: Tensor[*Ts] | float) -> Tensor[*Ts]:
        return self._binary(other, operator.add)

    def __radd__(self, other: float) -> Tensor[*Ts]:
        return self._binary(other, operator.add, reflected=True)

    def __sub__(self, other: Tensor[*Ts] | float) -> Tensor[*Ts]:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: float) -> Tensor[*Ts]:
        return self._binary(other, operator.sub, reflected=True)

    def __mul__(self, other: Tensor[*Ts] | float) -> Tensor[*Ts]:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: float) -> Tensor[*Ts]:
        return self._binary(other, operator.mul, reflected=True)

    def __truediv__(self, other: Tensor[*Ts] | float) -> Tensor[*Ts]:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: float) -> Tensor[*Ts]:
        return self._binary(other, operator.truediv, reflected=True)

    def __repr__(self) -> str:
        parts = ", ".join(f"{_dim_name(d)}={s}" for d, s in zip(self.dims, self.shape))
        return f"Tensor[{parts}]"


def _broadcast_dims(a: Tensor[*tuple[Any, ...]], b: Tensor[*tuple[Any, ...]]) -> tuple[Any, ...]:
    if len(a.dims) != len(b.dims):
        raise DimError(f"cannot broadcast {_fmt(a.dims)} with {_fmt(b.dims)}: rank differs")
    out = []
    for da, db, sa, sb in zip(a.dims, b.dims, a.shape, b.shape):
        if not _compatible(da, db):
            raise DimError(f"cannot broadcast {_fmt(a.dims)} with {_fmt(b.dims)}")
        if sa != sb and 1 not in (sa, sb):
            raise DimError(
                f"cannot broadcast {_dim_name(da)}={sa} with {_dim_name(db)}={sb}"
            )
        out.append(db if _is_identity(da) else da)
    return tuple(out)


AnyTensor: TypeAlias = "Tensor[*tuple[Any, ...]]"
T = TypeVar("T", bound=AnyTensor)
U = TypeVar("U", bound=AnyTensor)
P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Backend-agnostic array helpers
# ---------------------------------------------------------------------------


def _shape(data: Any) -> tuple[int, ...]:
    return tuple(data.shape)


def _check_rank(data: Any, dims: tuple[Any, ...]) -> None:
    shape = _shape(data)
    if len(shape) != len(dims):
        raise DimError(
            f"array of rank {len(shape)} (shape {shape}) cannot have dims {_fmt(dims)}"
        )


def _transpose(data: Any, axes: tuple[int, ...]) -> Any:
    if axes == tuple(range(len(axes))):
        return data
    # torch.Tensor.transpose only swaps two axes, so prefer .permute when it exists.
    if hasattr(data, "permute"):
        return data.permute(*axes)
    return data.transpose(axes)


# ---------------------------------------------------------------------------
# Type introspection
# ---------------------------------------------------------------------------


def dims_of(tp: Any) -> tuple[Any, ...]:
    """Runtime dims of a tensor type, e.g. ``dims_of(Audio) == (Batch, Channels, Time)``."""
    if get_origin(tp) is not Tensor:
        raise TypeError(f"expected a parameterized Tensor type, got {tp!r}")
    return get_args(tp)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def typed(tp: type[T], data: Any) -> T:
    """Attach dimension types to a raw array: ``x = typed(Audio, np.zeros((8, 2, 16000)))``."""
    dims = dims_of(tp)
    _check_rank(data, dims)
    return cast(T, Tensor(data, dims))


def check(x: AnyTensor, tp: type[T]) -> T:
    """Runtime-verify that ``x`` fits ``tp`` (Identity matches anything), then narrow to it."""
    want = dims_of(tp)
    if len(want) != len(x.dims) or not all(map(_compatible, x.dims, want)):
        raise DimError(f"expected {_fmt(want)}, got {_fmt(x.dims)}")
    return cast(T, x)


def _plan_permutation(
    src: tuple[Any, ...], dst: tuple[Any, ...]
) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    """
    Work out the axis order that takes ``src`` to ``dst``.

    1. Named dims are matched by identity. Repeated dims (e.g. ``[Time, Time]``)
       match in order of appearance.
    2. Leftover target slots are filled from leftover source axes, in order,
       but only where one side is Identity.

    Returns ``(axes, result_dims)``. ``result_dims`` keeps the more specific dim
    whenever an Identity slot was matched against a named one, so the runtime
    never loses information it had.
    """
    if len(src) != len(dst):
        raise DimError(f"cannot permute {_fmt(src)} into {_fmt(dst)}: rank differs")

    available: dict[Any, deque[int]] = {}
    for i, d in enumerate(src):
        if not _is_identity(d):
            available.setdefault(d, deque()).append(i)

    axes: list[int | None] = [None] * len(dst)
    used: set[int] = set()
    for j, d in enumerate(dst):
        if not _is_identity(d) and available.get(d):
            i = available[d].popleft()
            axes[j] = i
            used.add(i)

    leftover = deque(i for i in range(len(src)) if i not in used)
    result = list(dst)
    for j, d in enumerate(dst):
        if axes[j] is not None:
            continue
        i = leftover.popleft()
        if not _compatible(src[i], d):
            raise DimError(
                f"cannot permute {_fmt(src)} into {_fmt(dst)}: "
                f"no source axis for {_dim_name(d)}"
            )
        axes[j] = i
        if _is_identity(d):
            result[j] = src[i]

    return tuple(cast(list[int], axes)), tuple(result)


def _permute_dims(x: AnyTensor, dst: tuple[Any, ...]) -> AnyTensor:
    axes, result = _plan_permutation(x.dims, dst)
    return Tensor(_transpose(x.data, axes), result)


def permute(x: AnyTensor, to: type[T]) -> T:
    """
    Rearrange ``x`` into the layout of ``to``. The axis order comes from the types::

        y = permute(spec, to=Tensor[Batch, Channels, Time, Frequency])
    """
    return cast(T, _permute_dims(x, dims_of(to)))


def relabel(x: AnyTensor, to: type[T]) -> T:
    """
    Reinterpret the axes of ``x`` as ``to`` without moving data (rank must match).

    This is the explicit escape hatch for an operation that changes what an axis
    *means* but not where it is, e.g. a conv that maps Channels -> Features.
    """
    dims = dims_of(to)
    _check_rank(x.data, dims)
    return cast(T, Tensor(x.data, dims))


def transform(x: AnyTensor, fn: Callable[[Any], Any], to: type[T]) -> T:
    """
    Apply a raw-array function that produces a *new* tensor type, e.g. an STFT
    that turns Audio into a Spectrogram. The result's rank is checked against ``to``.
    """
    return typed(to, fn(x.data))


def via(x: T, layout: type[U], fn: Callable[[U], U]) -> T:
    """
    Run a typed function that wants a specific layout, then restore the caller's layout::

        y: Spectrogram = via(spec, SpecFreqLast, normalize_over_freq)

    ``x`` is permuted into ``layout``, ``fn`` is applied, and the result is permuted
    back into ``x``'s original layout. Sizes may change; layout may not.
    """
    y = fn(permute(x, layout))
    return cast(T, _permute_dims(check(y, layout), x.dims))


def via_raw(x: T, layout: type[AnyTensor], fn: Callable[[Any], Any]) -> T:
    """
    Like ``via``, but ``fn`` takes and returns raw arrays. Use it to call untyped
    library code (``torch.nn.functional.conv1d``, ``jnp.fft.rfft``, ...) that
    expects a particular axis order.
    """
    inner = permute(x, layout)
    out: AnyTensor = Tensor(fn(inner.data), inner.dims)
    _check_rank(out.data, out.dims)
    return cast(T, _permute_dims(out, x.dims))


# ---------------------------------------------------------------------------
# Runtime checking of annotated functions
# ---------------------------------------------------------------------------


def checked(fn: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that verifies every ``Tensor[...]``-annotated argument and the return
    value against their annotations at call time. It complements the static
    checker at boundaries where types were lost (``Any``, untyped libraries, ...).
    """
    sig = inspect.signature(fn)
    hints: dict[str, Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal hints
        if hints is None:  # resolve lazily so forward references work
            hints = get_type_hints(fn)
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            tp: Any = hints.get(name)
            if get_origin(tp) is Tensor:
                if not isinstance(value, Tensor):
                    raise DimError(f"{fn.__name__}({name}=...): expected Tensor, got {type(value)}")
                try:
                    check(value, tp)
                except DimError as e:
                    raise DimError(f"{fn.__name__}({name}=...): {e}") from None
        result = fn(*args, **kwargs)
        ret: Any = hints.get("return")
        if get_origin(ret) is Tensor:
            try:
                check(cast(AnyTensor, result), ret)
            except DimError as e:
                raise DimError(f"{fn.__name__} -> {e}") from None
        return result

    return wrapper


# ===========================================================================
# Examples
# ===========================================================================
#
# These use a tiny shape-only stand-in for numpy/torch so the file stays
# stdlib-only. With real arrays nothing changes: typed(Audio, torch.randn(8, 2, 16000)).

if __name__ == "__main__":

    class ShapeArray:
        """Minimal 'meta' array: has a shape and supports numpy-style transpose."""

        def __init__(self, *shape: int) -> None:
            self.shape = tuple(shape)

        def transpose(self, axes: tuple[int, ...]) -> ShapeArray:
            return ShapeArray(*(self.shape[a] for a in axes))

        def _bcast(self, other: object) -> ShapeArray:
            if not isinstance(other, ShapeArray):
                return self
            return ShapeArray(*(max(a, b) for a, b in zip(self.shape, other.shape)))

        __add__ = __radd__ = __sub__ = __rsub__ = _bcast
        __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _bcast

        def __repr__(self) -> str:
            return f"ShapeArray{self.shape}"

    # --- 1. Declare the vocabulary of the experiment ----------------------

    Batch = NewType("Batch", Dim)
    Channels = NewType("Channels", Dim)
    Time = NewType("Time", Dim)
    Frequency = NewType("Frequency", Dim)
    Features = NewType("Features", Dim)

    Audio = Tensor[Batch, Channels, Time]
    Spectrogram = Tensor[Batch, Frequency, Channels, Time]
    SpecFreqLast = Tensor[Batch, Channels, Time, Frequency]
    FrameFeatures = Tensor[Batch, Features, Time]

    # --- 2. A typed boundary: Audio -> Spectrogram ------------------------

    @checked
    def stft(audio: Audio, n_freq: int = 513, hop: int = 256) -> Spectrogram:
        def fake_stft(a: ShapeArray) -> ShapeArray:
            b, c, t = a.shape
            return ShapeArray(b, n_freq, c, t // hop)

        return transform(audio, fake_stft, to=Spectrogram)

    # --- 3. A function that only works in one layout ----------------------

    def normalize_over_freq(spec: SpecFreqLast) -> SpecFreqLast:
        # imagine: spec.map(lambda d: d / d.norm(dim=-1, keepdim=True))
        return spec.map(lambda d: d)

    # --- 4. Calling untyped library code that wants a given layout ----------

    def encode(spec: Spectrogram) -> FrameFeatures:
        # Fold (Frequency, Channels) into Features, as conv1d over time expects.
        def fold(d: ShapeArray) -> ShapeArray:
            b, f, c, t = d.shape  # imagine: d.reshape(b, f * c, t)
            return ShapeArray(b, f * c, t)

        return transform(spec, fold, to=FrameFeatures)

    # --- Run it ---------------------------------------------------------------

    audio = typed(Audio, ShapeArray(8, 2, 16000))
    print("audio:          ", audio)

    spec = stft(audio)
    print("spec:           ", spec)

    # Explicit permutation: the axis order is derived from the types.
    spec_fl = permute(spec, to=SpecFreqLast)
    print("freq-last:      ", spec_fl)

    # Permute in, apply, permute back. The result is still a Spectrogram.
    normed: Spectrogram = via(spec, SpecFreqLast, normalize_over_freq)
    print("normalized:     ", normed)

    # Raw library call that wants channels-last audio.
    def needs_channels_last(d: ShapeArray) -> ShapeArray:
        return d  # imagine: some_effect(d) with d: [B, T, C]

    fx: Audio = via_raw(audio, Tensor[Batch, Time, Channels], needs_channels_last)
    print("fx (round trip):", fx)

    feats = encode(spec)
    print("features:       ", feats, "| Features size =", feats.size(Features))

    # Identity is a wildcard: generic utilities need only the dims they touch.
    def time_last_frames(x: Tensor[Identity, Identity, Time]) -> int:
        return x.shape[-1]

    print("frames (audio):  ", time_last_frames(audio))
    print("frames (feats):  ", time_last_frames(feats))

    # Permuting into a target with Identity slots keeps the specific runtime dims.
    loose = permute(spec, to=Tensor[Identity, Identity, Time, Frequency])
    print("loose permute:  ", loose)

    # Broadcasting: a per-(channel, time) gain shared across the batch.
    Gain = Tensor[Identity, Channels, Time]
    gain = typed(Gain, ShapeArray(1, 2, 16000))
    scaled: Audio = audio * gain
    print("audio * gain:   ", scaled)
    print("0.5 * audio:    ", 0.5 * audio)

    # --- Runtime failures (always caught, even without a type checker) --------

    bad_gain = typed(Gain, ShapeArray(1, 3, 16000))

    thunks: list[tuple[str, Callable[[], object]]] = [
        ("broadcast size", lambda: audio * bad_gain),
        ("wrong rank", lambda: typed(Audio, ShapeArray(8, 16000))),
        ("bad permute", lambda: permute(audio, to=Tensor[Batch, Frequency, Time])),
        ("checked arg", lambda: stft(cast(Audio, spec))),
    ]
    for label, thunk in thunks:
        try:
            thunk()
        except DimError as e:
            print(f"DimError ({label}): {e}")

    # --- Static failures (uncomment and run pyright / mypy) -------------------
    #
    # stft(spec)                        # Spectrogram is not Audio
    # normalize_over_freq(spec)         # [B, F, C, T] is not [B, C, T, F]
    # bad: Audio = permute(audio, to=Tensor[Batch, Time, Channels])
    # time_last_frames(spec_fl)         # 4 dims, not 3
    # audio * typed(Tensor[Batch, Frequency, Time], ...)   # Frequency vs Channels
