"""
tensor_einsum.py: prototype of inferring einsum calls from tensor type annotations.

Builds on tensor_typing.py and, like it, imports only from the standard library.
The einsum implementation is taken from whichever backend the operands come from
(numpy, torch or jax), so no array library is imported here.

The idea
--------
Every dimension is a nominal type, so a function signature already says everything
einsum needs::

    @einsum_contract
    def mel_project(
        spec: Tensor[Batch, Frequency, Channels, Time],
        fb: Tensor[Mel, Frequency],
    ) -> Tensor[Batch, Mel, Channels, Time]: ...

    mel_project.einsum_spec  # 'abcd,eb->aecd'

The rules are the usual einsum ones, read off the types:

* the same dimension type in several inputs is the same index;
* a dimension that appears in an input but not in the output is summed over;
* the output order is the order in the return annotation, so einsum also does
  any permutation for free.

The static checker sees an ordinary typed function, so call sites are checked
exactly as before. The runtime checks the operands against the annotations,
checks that shared dims have equal sizes, and dispatches to the backend's einsum.

There is also an inline form, which takes its input dims from the runtime tensors::

    y = einsum(spec, fb, to=Tensor[Batch, Mel, Channels, Time])
"""

from __future__ import annotations

import functools
import inspect
import string
import sys
from typing import Any, Callable, ParamSpec, Protocol, TypeVar, cast, get_origin, get_type_hints

from tensor_typing import (
    AnyTensor,
    DimError,
    Tensor,
    _check_consistent,
    _dim_name,
    _fmt,
    _is_identity,
    check,
    dims_of,
)

__all__ = ["EinsumFunction", "einsum", "einsum_contract", "einsum_spec"]

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=AnyTensor)
P_co = ParamSpec("P_co")
R_co = TypeVar("R_co", covariant=True)


class EinsumFunction(Protocol[P_co, R_co]):
    """What `einsum_contract` returns: the original signature plus `.einsum_spec`."""

    __name__: str
    einsum_spec: str

    def __call__(self, *args: P_co.args, **kwargs: P_co.kwargs) -> R_co: ...

_LETTERS = string.ascii_letters


# ---------------------------------------------------------------------------
# Types -> einsum string
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def einsum_spec(inputs: tuple[tuple[Any, ...], ...], output: tuple[Any, ...]) -> str:
    """
    Build an einsum string from input and output dim tuples::

        einsum_spec(((Batch, Channels, Time), (Sources, Channels)), (Batch, Sources, Time))
        # 'abc,db->adc'
    """
    letters: dict[Any, str] = {}

    def letter(d: Any) -> str:
        if _is_identity(d):
            raise DimError(
                "Identity dims have no name, so einsum can't match them across "
                "operands. Use a named dim (or `relabel`) instead."
            )
        if d not in letters:
            if len(letters) == len(_LETTERS):
                raise DimError(f"einsum supports at most {len(_LETTERS)} distinct dims")
            letters[d] = _LETTERS[len(letters)]
        return letters[d]

    for dims in (*inputs, output):
        if len(set(dims)) != len(dims):
            raise DimError(
                f"{_fmt(dims)} repeats a dim. In einsum that means a diagonal, which is "
                "rarely what's meant; give each axis its own dim type (e.g. Time and Lag)."
            )

    ins = ["".join(letter(d) for d in dims) for dims in inputs]

    for d in output:
        if _is_identity(d):
            raise DimError("the output of an einsum can't contain Identity dims")
        if d not in letters:
            raise DimError(f"output dim {_dim_name(d)} doesn't appear in any input")
    out = "".join(letters[d] for d in output)

    return ",".join(ins) + "->" + out


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


def _backend_einsum(data: Any) -> Callable[..., Any]:
    """Find the einsum of the library `data` belongs to, without importing it here."""
    root = type(data).__module__.split(".")[0]
    if root == "torch":
        return cast(Callable[..., Any], sys.modules["torch"].einsum)
    if root in ("jax", "jaxlib"):  # concrete arrays and tracers under jit/vmap
        return cast(Callable[..., Any], sys.modules["jax.numpy"].einsum)
    if root == "numpy":
        return cast(Callable[..., Any], sys.modules["numpy"].einsum)
    raise TypeError(f"don't know which einsum to use for {type(data)!r}")


def _check_sizes(operands: tuple[AnyTensor, ...]) -> None:
    sizes: dict[Any, int] = {}
    for x in operands:
        for d, s in zip(x.dims, x.shape):
            if d in sizes and sizes[d] != s:
                raise DimError(f"dim {_dim_name(d)} has size {sizes[d]} in one operand and {s} in another")
            sizes[d] = s


def _einsum(operands: tuple[AnyTensor, ...], output: tuple[Any, ...]) -> AnyTensor:
    if not operands:
        raise TypeError("einsum needs at least one operand")
    for x in operands:
        _check_consistent(x)
    spec = einsum_spec(tuple(x.dims for x in operands), output)
    _check_sizes(operands)
    fn = _backend_einsum(operands[0].data)
    return Tensor(fn(spec, *(x.data for x in operands)), output)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def einsum(*operands: AnyTensor, to: type[T]) -> T:
    """
    Inline form: contract the operands into the layout of `to`::

        y = einsum(x, w, to=Tensor[Batch, Sources, Time])

    Input dims come from the runtime tensors, so the static checker verifies only
    the result type. Use `einsum_contract` to have the inputs checked statically too.
    """
    return cast(T, _einsum(operands, dims_of(to)))


def einsum_contract(fn: Callable[P, R]) -> EinsumFunction[P, R]:
    """
    Turn a typed stub into an einsum. Every parameter must be annotated with a
    `Tensor[...]` type; the body is never run (write `...`). Note that mypy reports
    `...` bodies as `empty-body` errors in .py files; pyright accepts them.

    The einsum string is built and validated when the function is defined, if the
    annotations can be resolved then; otherwise on the first call. It is exposed as
    `fn.einsum_spec` for inspection.
    """
    sig = inspect.signature(fn)

    def resolve() -> tuple[list[tuple[str, Any]], tuple[Any, ...], str]:
        hints = get_type_hints(fn)
        params: list[tuple[str, Any]] = []
        for name in sig.parameters:
            tp = hints.get(name)
            if get_origin(tp) is not Tensor:
                raise TypeError(f"{fn.__name__}: parameter {name!r} must be annotated with Tensor[...]")
            params.append((name, tp))
        ret = hints.get("return")
        if get_origin(ret) is not Tensor:
            raise TypeError(f"{fn.__name__}: return must be annotated with Tensor[...]")
        out = dims_of(ret)
        # Identity inputs are resolved per call from the runtime dims, so only a
        # fully named signature can be validated up front.
        if all(not _is_identity(d) for _, tp in params for d in dims_of(tp)):
            spec = einsum_spec(tuple(dims_of(tp) for _, tp in params), out)
        else:
            spec = "<resolved per call>"
        return params, out, spec

    state: dict[str, Any] = {}
    try:
        state["params"], state["out"], state["spec"] = resolve()
    except NameError:
        pass  # forward references; resolve on first call

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not state:
            state["params"], state["out"], state["spec"] = resolve()
            result.einsum_spec = state["spec"]
        bound = sig.bind(*args, **kwargs)
        operands: list[AnyTensor] = []
        for name, tp in state["params"]:
            value = bound.arguments[name]
            if not isinstance(value, Tensor):
                raise DimError(f"{fn.__name__}({name}=...): expected Tensor, got {type(value)}")
            try:
                operands.append(check(value, tp))
            except DimError as e:
                raise DimError(f"{fn.__name__}({name}=...): {e}") from None
        return cast(R, _einsum(tuple(operands), state["out"]))

    result = cast(EinsumFunction[P, R], wrapper)
    result.einsum_spec = state.get("spec", "<resolved on first call>")
    return result


# ===========================================================================
# Examples (need jax; run with JAX_PLATFORMS=cpu to skip GPU discovery)
# ===========================================================================

if __name__ == "__main__":
    from typing import NewType

    import jax
    import jax.numpy as jnp

    from tensor_typing import Dim, Identity, typed

    Batch = NewType("Batch", Dim)
    Channels = NewType("Channels", Dim)
    Time = NewType("Time", Dim)
    Frequency = NewType("Frequency", Dim)
    Mel = NewType("Mel", Dim)
    Sources = NewType("Sources", Dim)
    Atom = NewType("Atom", Dim)

    Audio = Tensor[Batch, Channels, Time]
    Spectrogram = Tensor[Batch, Frequency, Channels, Time]
    Dictionary = Tensor[Atom, Time]

    # Projection onto a filterbank: Frequency is contracted.
    @einsum_contract
    def mel_project(spec: Spectrogram, fb: Tensor[Mel, Frequency]) -> Tensor[Batch, Mel, Channels, Time]: ...

    # Channel unmixing: Channels -> Sources.
    @einsum_contract
    def unmix(x: Audio, w: Tensor[Sources, Channels]) -> Tensor[Batch, Sources, Time]: ...

    # Matching pursuit: inner product of each atom with each channel (Time contracted)...
    @einsum_contract
    def atom_scores(x: Audio, atoms: Dictionary) -> Tensor[Batch, Channels, Atom]: ...

    # ...and resynthesis from coefficients (Atom contracted).
    @einsum_contract
    def synthesize(coeffs: Tensor[Batch, Channels, Atom], atoms: Dictionary) -> Audio: ...

    # One operand: reduction plus permutation.
    @einsum_contract
    def sum_over_freq_time_major(spec: Spectrogram) -> Tensor[Time, Batch, Channels]: ...

    # Identity in an input: resolved from the runtime dims at each call.
    @einsum_contract
    def mono(x: Tensor[Identity, Channels, Time]) -> Tensor[Batch, Time]: ...

    for f in (mel_project, unmix, atom_scores, synthesize, sum_over_freq_time_major, mono):
        print(f"{f.__name__:26s} {f.einsum_spec}")
    print()

    key = jax.random.split(jax.random.PRNGKey(0), 5)
    nb, nc, nt, nf, nm, ns, na = 4, 2, 1024, 257, 64, 3, 16
    audio = typed(Audio, jax.random.normal(key[0], (nb, nc, nt)))
    spec = typed(Spectrogram, jax.random.normal(key[1], (nb, nf, nc, 32)))
    fb = typed(Tensor[Mel, Frequency], jax.random.normal(key[2], (nm, nf)))
    w = typed(Tensor[Sources, Channels], jax.random.normal(key[3], (ns, nc)))
    atoms = typed(Dictionary, jax.random.normal(key[4], (na, nt)))

    mel = mel_project(spec, fb)
    print("mel_project:    ", mel)
    assert jnp.allclose(mel.data, jnp.einsum("bfct,mf->bmct", spec.data, fb.data))

    print("unmix:          ", unmix(audio, w))

    scores = atom_scores(audio, atoms)
    recon = synthesize(scores, atoms)
    print("atom_scores:    ", scores)
    print("synthesize:     ", recon)

    print("sum/permute:    ", sum_over_freq_time_major(spec))
    print("mono:           ", mono(audio))

    # Inline form, no stub needed.
    print("inline einsum:  ", einsum(audio, w, to=Tensor[Sources, Batch, Time]))

    # Works under jit: the spec is computed at trace time from the (static) dims.
    @jax.jit
    def jitted(x: Any, a: Any) -> Any:
        return synthesize(atom_scores(typed(Audio, x), typed(Dictionary, a)), typed(Dictionary, a)).data

    assert jnp.allclose(jitted(audio.data, atoms.data), recon.data, atol=1e-3)
    print("jit:             ok")
    print()

    # Errors
    def bad_size() -> object:
        return unmix(audio, typed(Tensor[Sources, Channels], jnp.ones((ns, 5))))

    def output_not_in_inputs() -> object:
        @einsum_contract
        def oops(x: Audio) -> Tensor[Batch, Mel]: ...

        return oops

    def gram() -> object:
        @einsum_contract
        def gram(x: Audio) -> Tensor[Batch, Time, Time]: ...

        return gram

    for thunk in (bad_size, output_not_in_inputs, gram):
        try:
            thunk()
        except DimError as e:
            print(f"DimError ({thunk.__name__}): {e}")
