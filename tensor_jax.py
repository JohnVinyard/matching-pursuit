"""
tensor_jax.py: JAX integration for tensor_typing.

Importing this module registers ``Tensor`` as a JAX pytree. The raw array is the
single leaf and the dims tuple is static auxiliary data. After that:

* ``jax.jit`` accepts and returns Tensors. Dims are part of the cache key, so the
  same shapes with different dims trace separately.
* ``jax.grad`` / ``jax.value_and_grad`` differentiate with respect to Tensors, and
  the gradient is a Tensor with the same dims.
* Tensors can live inside equinox modules, optax states and any other pytree.

vmap
----
Plain ``jax.vmap`` can't be used on Tensors directly. It strips an axis from the
data, but the pytree can't tell which one, so the dims inside the mapped function
would be stale. (Core ops raise a clear DimError if that happens.) Use the typed
``vmap`` here instead, which maps over a *named* dim::

    per_example = vmap(fn, over=Batch)

Every Tensor argument containing ``Batch`` is mapped over that axis, wherever it is.
Tensor arguments without it are broadcast. Inside ``fn`` the dims no longer
contain ``Batch``, and ``Batch`` is put back as the leading dim of every Tensor in
the output.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, TypeVarTuple, overload

import jax

from tensor_typing import DimError, Tensor, _dim_name

__all__ = ["vmap"]


# ---------------------------------------------------------------------------
# Pytree registration
# ---------------------------------------------------------------------------


def _flatten_with_keys(x: Tensor[*tuple[Any, ...]]) -> tuple[tuple[tuple[Any, Any], ...], tuple[Any, ...]]:
    return ((jax.tree_util.GetAttrKey("data"), x.data),), x.dims


def _flatten(x: Tensor[*tuple[Any, ...]]) -> tuple[tuple[Any], tuple[Any, ...]]:
    return (x.data,), x.dims


def _unflatten(dims: tuple[Any, ...], children: Any) -> Tensor[*tuple[Any, ...]]:
    # JAX also unflattens with non-array leaves (None, in_axes ints, sentinels),
    # so no validation here.
    (data,) = children
    return Tensor(data, dims)


try:
    jax.tree_util.register_pytree_with_keys(Tensor, _flatten_with_keys, _unflatten, _flatten)
except ValueError:
    pass  # already registered (e.g. module reloaded in a notebook)


# ---------------------------------------------------------------------------
# Typed vmap over a named dim
# ---------------------------------------------------------------------------

D = TypeVar("D")
Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")


def _is_tensor(x: Any) -> bool:
    return isinstance(x, Tensor)


@overload
def vmap(
    fn: Callable[[Tensor[*Ts]], Tensor[*Us]], over: Callable[[Any], D]
) -> Callable[[Tensor[D, *Ts]], Tensor[D, *Us]]: ...


@overload
def vmap(fn: Callable[..., Any], over: Any) -> Callable[..., Any]: ...


def vmap(fn: Callable[..., Any], over: Any) -> Callable[..., Any]:
    """
    Map ``fn`` over the named dim ``over``.

    Typing: a single-Tensor function ``Tensor[*Ts] -> Tensor[*Us]`` becomes
    ``Tensor[over, *Ts] -> Tensor[over, *Us]``. The static type assumes ``over``
    is the leading dim of the input, but at runtime it can be at any position.
    Functions with other signatures are typed ``Callable[..., Any]``.

    Positional args are mapped or broadcast as described in the module docstring.
    Keyword args are always broadcast.
    """

    @functools.wraps(fn)
    def mapped(*args: Any, **kwargs: Any) -> Any:
        leaves, treedef = jax.tree_util.tree_flatten(args, is_leaf=_is_tensor)

        in_axes: list[int | None] = []
        raw: list[Any] = []
        inner_dims: list[tuple[Any, ...] | None] = []  # None marks a non-Tensor leaf
        for leaf in leaves:
            if isinstance(leaf, Tensor):
                raw.append(leaf.data)
                if over in leaf.dims:
                    ax = leaf.axis(over)
                    in_axes.append(ax)
                    inner_dims.append(leaf.dims[:ax] + leaf.dims[ax + 1 :])
                else:
                    in_axes.append(None)
                    inner_dims.append(leaf.dims)
            else:
                raw.append(leaf)
                in_axes.append(None)
                inner_dims.append(None)

        if all(a is None for a in in_axes):
            raise DimError(f"vmap(over={_dim_name(over)}): no Tensor argument has that dim")

        out_meta: dict[str, Any] = {}

        def inner(*raw_leaves: Any) -> list[Any]:
            rebuilt = [
                r if dims is None else Tensor(r, dims)
                for r, dims in zip(raw_leaves, inner_dims)
            ]
            out = fn(*jax.tree_util.tree_unflatten(treedef, rebuilt), **kwargs)
            out_leaves, out_def = jax.tree_util.tree_flatten(out, is_leaf=_is_tensor)
            out_meta["treedef"] = out_def
            out_meta["dims"] = [l.dims if isinstance(l, Tensor) else None for l in out_leaves]
            for l in out_leaves:
                if isinstance(l, Tensor) and over in l.dims:
                    raise DimError(
                        f"vmap(over={_dim_name(over)}): output already has that dim "
                        f"{l.dims!r}; it is added back automatically"
                    )
            return [l.data if isinstance(l, Tensor) else l for l in out_leaves]

        raw_out = jax.vmap(inner, in_axes=tuple(in_axes))(*raw)
        restored = [
            r if dims is None else Tensor(r, (over, *dims))
            for r, dims in zip(raw_out, out_meta["dims"])
        ]
        return jax.tree_util.tree_unflatten(out_meta["treedef"], restored)

    return mapped


# ===========================================================================
# Examples (run with JAX_PLATFORMS=cpu to skip GPU discovery)
# ===========================================================================

if __name__ == "__main__":
    from typing import NewType

    import jax.numpy as jnp

    from tensor_einsum import einsum_contract
    from tensor_typing import Dim, permute, typed

    Batch = NewType("Batch", Dim)
    Channels = NewType("Channels", Dim)
    Time = NewType("Time", Dim)
    Atom = NewType("Atom", Dim)

    Audio = Tensor[Batch, Channels, Time]
    Dictionary = Tensor[Atom, Time]

    @einsum_contract
    def atom_scores(x: Audio, atoms: Dictionary) -> Tensor[Batch, Channels, Atom]: ...

    @einsum_contract
    def synthesize(coeffs: Tensor[Batch, Channels, Atom], atoms: Dictionary) -> Audio: ...

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    nb, nc, nt, na = 4, 2, 256, 8
    audio = typed(Audio, jax.random.normal(k1, (nb, nc, nt)))
    atoms = typed(Dictionary, jax.random.normal(k2, (na, nt)) / jnp.sqrt(nt))

    # --- jit: Tensors in, Tensors out ----------------------------------------
    @jax.jit
    def analyse_resynth(x: Audio, d: Dictionary) -> Audio:
        return synthesize(atom_scores(x, d), d)

    recon = analyse_resynth(audio, atoms)
    print("jit:   ", recon)

    # --- grad: the gradient is a Tensor with the same dims ---------------------
    def loss(d: Dictionary, x: Audio) -> jax.Array:
        r = x - synthesize(atom_scores(x, d), d)
        return jnp.mean(r.data**2)

    g = jax.jit(jax.grad(loss))(atoms, audio)
    print("grad:  ", g)
    assert g.dims == atoms.dims

    # A few steps of gradient descent with tree_map over Tensors.
    d = atoms
    for _ in range(50):
        g = jax.grad(loss)(d, audio)
        d = jax.tree_util.tree_map(lambda p, gp: p - 0.5 * gp, d, g)
    print(f"loss:   {float(loss(atoms, audio)):.4f} -> {float(loss(d, audio)):.4f}")

    # --- vmap over a named dim ------------------------------------------------
    PerExample = Tensor[Channels, Time]

    def peak_normalize(x: PerExample) -> PerExample:
        return x.map(lambda d: d / jnp.max(jnp.abs(d)))

    batched = vmap(peak_normalize, over=Batch)
    y = batched(audio)
    print("vmap:  ", y)
    assert jnp.allclose(y.data, audio.data / jnp.max(jnp.abs(audio.data), axis=(1, 2), keepdims=True))

    # Batch need not be leading: the axis is found by name.
    time_major = permute(audio, to=Tensor[Time, Batch, Channels])
    print("vmap (Batch at axis 1):", vmap(peak_normalize, over=Batch)(time_major))  # type: ignore[arg-type]

    # Multiple args: Tensors without the dim (the dictionary) are broadcast.
    def per_example_scores(x: PerExample, d: Dictionary) -> Tensor[Channels, Atom]:
        return typed(Tensor[Channels, Atom], jnp.einsum("ct,at->ca", x.data, d.data))

    s = vmap(per_example_scores, over=Batch)(audio, atoms)
    print("vmap 2:", s)
    assert jnp.allclose(s.data, atom_scores(audio, atoms).data, atol=1e-5)

    # vmap inside grad inside jit.
    def batched_loss(d: Dictionary, x: Audio) -> jax.Array:
        def one(xi: PerExample, d: Dictionary) -> jax.Array:
            c = typed(Tensor[Channels, Atom], jnp.einsum("ct,at->ca", xi.data, d.data))
            r = xi.data - jnp.einsum("ca,at->ct", c.data, d.data)
            return jnp.mean(r**2)

        return jnp.mean(vmap(one, over=Batch)(x, d))

    g2 = jax.jit(jax.grad(batched_loss))(atoms, audio)
    print("jit(grad(vmap)):", g2, "| matches unbatched grad:", bool(jnp.allclose(g2.data, jax.grad(loss)(atoms, audio).data, atol=1e-5)))

    # --- What plain jax.vmap does ---------------------------------------------
    try:
        jax.vmap(lambda x: permute(x, to=Tensor[Batch, Time, Channels]))(audio)
    except DimError as e:
        print("plain jax.vmap ->", e)
