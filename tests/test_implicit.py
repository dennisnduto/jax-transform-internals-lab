import jax
import jax.numpy as jnp
import numpy as np

from jax_transform_internals import implicit_tanh, init_contractive_params, unrolled_tanh


def _problem():
    params = init_contractive_params(jax.random.key(0), input_size=3, hidden_size=4)
    x = jnp.array([0.2, -0.1, 0.4], dtype=jnp.float32)
    return params, x


def test_primal_matches_long_unrolled_reference():
    params, x = _problem()
    custom = implicit_tanh(params, x)
    reference = unrolled_tanh(params, x, steps=250)
    np.testing.assert_allclose(custom, reference, rtol=2e-5, atol=2e-6)


def test_custom_vjp_gradient_matches_unrolled_reference():
    params, x = _problem()

    custom_grad = jax.grad(lambda p: jnp.sum(implicit_tanh(p, x) ** 2))(params)
    reference_grad = jax.grad(lambda p: jnp.sum(unrolled_tanh(p, x, steps=250) ** 2))(params)

    for actual, expected in zip(custom_grad, reference_grad, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)


def test_one_parameter_matches_finite_difference():
    params, x = _problem()

    def loss(p):
        return jnp.sum(implicit_tanh(p, x) ** 2)

    gradient = jax.grad(loss)(params)

    epsilon = 1e-3
    plus = params._replace(recurrent=params.recurrent.at[0, 0].add(epsilon))
    minus = params._replace(recurrent=params.recurrent.at[0, 0].add(-epsilon))
    finite_difference = (loss(plus) - loss(minus)) / (2 * epsilon)

    np.testing.assert_allclose(gradient.recurrent[0, 0], finite_difference, rtol=2e-2, atol=2e-4)


def test_jit_vmap_and_reverse_mode_compose():
    params, x = _problem()
    batch = jnp.stack([x, 2 * x, -x])
    batched = jax.jit(jax.vmap(lambda row: implicit_tanh(params, row)))(batch)
    assert batched.shape == (3, 4)
    assert bool(jnp.all(jnp.isfinite(batched)))

    loss_and_grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(
        jax.vmap(lambda row: implicit_tanh(p, row))(batch) ** 2
    )))
    loss, gradient = loss_and_grad(params)
    assert bool(jnp.isfinite(loss))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradient))
