"""Compare implicit and unrolled gradients for a small equilibrium layer."""

import jax
import jax.numpy as jnp

from jax_transform_internals import implicit_tanh, init_contractive_params, unrolled_tanh

params = init_contractive_params(jax.random.key(11), input_size=3, hidden_size=4)
x = jnp.array([0.2, -0.1, 0.4], dtype=jnp.float32)

implicit_grad = jax.grad(lambda p: jnp.sum(implicit_tanh(p, x) ** 2))(params)
unrolled_grad = jax.grad(lambda p: jnp.sum(unrolled_tanh(p, x, 250) ** 2))(params)

for name, actual, expected in zip(params._fields, implicit_grad, unrolled_grad, strict=True):
    error = jnp.max(jnp.abs(actual - expected))
    print(f"{name:10s} max absolute gradient error: {float(error):.8e}")
