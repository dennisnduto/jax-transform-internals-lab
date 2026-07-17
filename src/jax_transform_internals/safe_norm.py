"""A custom JVP with an explicit derivative convention at the origin."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.custom_jvp
def safe_l2_norm(x: jax.Array) -> jax.Array:
    """Return the Euclidean norm and define a zero derivative at ``x == 0``.

    The primal is the exact L2 norm. The derivative of the L2 norm is not unique at the origin;
    this example chooses the zero vector as a deterministic, finite subgradient convention.
    """

    x = jnp.asarray(x)
    return jnp.sqrt(jnp.vdot(x, x).real)


@safe_l2_norm.defjvp
def _safe_l2_norm_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = safe_l2_norm(x)
    numerator = jnp.vdot(x, x_dot).real
    safe_denominator = jnp.where(primal_out > 0, primal_out, jnp.ones_like(primal_out))
    tangent_out = jnp.where(primal_out > 0, numerator / safe_denominator, 0.0)
    return primal_out, tangent_out


def safe_unit_vector(x: jax.Array) -> jax.Array:
    """Normalize a vector, returning zeros for the zero vector."""

    norm = safe_l2_norm(x)
    safe_denominator = jnp.where(norm > 0, norm, jnp.ones_like(norm))
    return jnp.where(norm > 0, x / safe_denominator, jnp.zeros_like(x))
