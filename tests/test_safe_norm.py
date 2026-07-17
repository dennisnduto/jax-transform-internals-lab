import jax
import jax.numpy as jnp
import numpy as np

from jax_transform_internals import safe_l2_norm, safe_unit_vector


def test_zero_gradient_is_finite_and_zero():
    zero = jnp.zeros((5,), dtype=jnp.float32)
    gradient = jax.grad(safe_l2_norm)(zero)
    np.testing.assert_allclose(gradient, np.zeros(5), atol=0.0)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_nonzero_gradient_matches_analytic_result():
    value = jnp.array([3.0, 4.0], dtype=jnp.float32)
    gradient = jax.grad(safe_l2_norm)(value)
    np.testing.assert_allclose(gradient, np.array([0.6, 0.8]), rtol=1e-6, atol=1e-6)


def test_jvp_jit_and_vmap_compose():
    batch = jnp.array([[3.0, 4.0], [0.0, 0.0], [5.0, 12.0]], dtype=jnp.float32)
    result = jax.jit(jax.vmap(safe_l2_norm))(batch)
    np.testing.assert_allclose(result, np.array([5.0, 0.0, 13.0]), rtol=1e-6)

    primal, tangent = jax.jvp(safe_l2_norm, (batch[0],), (jnp.ones(2),))
    np.testing.assert_allclose(primal, 5.0, rtol=1e-6)
    np.testing.assert_allclose(tangent, 1.4, rtol=1e-6)


def test_safe_unit_vector_handles_zero():
    result = safe_unit_vector(jnp.zeros((3,), dtype=jnp.float32))
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
