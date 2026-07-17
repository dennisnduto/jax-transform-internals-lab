"""Helpers for inspecting the staged representations produced by JAX."""

from __future__ import annotations

from collections.abc import Callable

import jax


def jaxpr_text(function: Callable, *example_args) -> str:
    """Return a textual JAXPR for ``function`` and concrete example arguments."""

    return str(jax.make_jaxpr(function)(*example_args))


def stablehlo_text(function: Callable, *example_args) -> str:
    """Lower a jitted function and return StableHLO text."""

    lowered = jax.jit(function).lower(*example_args)
    return str(lowered.compiler_ir(dialect="stablehlo"))
