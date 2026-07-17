"""Implicit differentiation through a contractive equilibrium layer."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres


class EquilibriumParams(NamedTuple):
    """Parameters for ``y = tanh(W @ y + U @ x + b)``."""

    recurrent: jax.Array
    input: jax.Array
    bias: jax.Array


def _update(params: EquilibriumParams, x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.tanh(params.recurrent @ y + params.input @ x + params.bias)


def _fixed_point(
    params: EquilibriumParams,
    x: jax.Array,
    max_steps: int | jax.Array,
    tolerance: float | jax.Array,
) -> jax.Array:
    hidden_size = params.recurrent.shape[0]
    dtype = jnp.result_type(params.recurrent, x)
    initial = jnp.zeros((hidden_size,), dtype=dtype)

    def condition(carry):
        step, _value, error = carry
        return (step < max_steps) & (error > tolerance)

    def body(carry):
        step, value, _error = carry
        next_value = _update(params, x, value)
        error = jnp.max(jnp.abs(next_value - value))
        return step + 1, next_value, error

    state = (
        jnp.array(0, dtype=jnp.int32),
        initial,
        jnp.array(jnp.inf, dtype=dtype),
    )
    return jax.lax.while_loop(condition, body, state)[1]


@jax.custom_vjp
def implicit_tanh(
    params: EquilibriumParams,
    x: jax.Array,
    max_steps: int = 80,
    tolerance: float = 1e-7,
) -> jax.Array:
    """Solve the equilibrium and differentiate it with the implicit function theorem."""

    return _fixed_point(params, x, max_steps, tolerance)


def _implicit_tanh_fwd(params, x, max_steps=80, tolerance=1e-7):
    solution = _fixed_point(params, x, max_steps, tolerance)
    residuals = (params, x, solution, max_steps, tolerance)
    return solution, residuals


def _implicit_tanh_bwd(residuals, output_cotangent):
    params, x, solution, max_steps, tolerance = residuals

    # Build the transpose-Jacobian action with a VJP. The adjoint operator is
    # (I - dF/dy)^T and is supplied matrix-free to GMRES.
    _, pullback_solution = jax.vjp(lambda y: _update(params, x, y), solution)

    def adjoint_operator(vector):
        return vector - pullback_solution(vector)[0]

    adjoint, _info = gmres(
        adjoint_operator,
        output_cotangent,
        tol=jnp.maximum(jnp.asarray(tolerance), 1e-6),
        atol=0.0,
        maxiter=max_steps,
    )

    _, pullback_inputs = jax.vjp(
        lambda p, input_value: _update(p, input_value, solution),
        params,
        x,
    )
    params_cotangent, x_cotangent = pullback_inputs(adjoint)
    return params_cotangent, x_cotangent, None, None


implicit_tanh.defvjp(_implicit_tanh_fwd, _implicit_tanh_bwd)


def unrolled_tanh(
    params: EquilibriumParams,
    x: jax.Array,
    steps: int = 200,
) -> jax.Array:
    """Reference implementation that differentiates through every fixed iteration."""

    initial = jnp.zeros((params.recurrent.shape[0],), dtype=jnp.result_type(x, params.recurrent))

    def body(_index, value):
        return _update(params, x, value)

    return jax.lax.fori_loop(0, steps, body, initial)


def init_contractive_params(
    key: jax.Array,
    input_size: int,
    hidden_size: int,
    recurrent_scale: float = 0.2,
) -> EquilibriumParams:
    """Create deterministic, usually contractive parameters for examples and tests."""

    recurrent_key, input_key, bias_key = jax.random.split(key, 3)
    recurrent = jax.random.normal(recurrent_key, (hidden_size, hidden_size))
    spectral_norm = jnp.linalg.svd(recurrent, compute_uv=False)[0]
    recurrent = recurrent * (recurrent_scale / jnp.maximum(spectral_norm, 1e-6))
    input_matrix = 0.25 * jax.random.normal(input_key, (hidden_size, input_size))
    bias = 0.05 * jax.random.normal(bias_key, (hidden_size,))
    return EquilibriumParams(recurrent, input_matrix, bias)
