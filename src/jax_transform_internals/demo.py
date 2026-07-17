"""Run a reproducible transformation-composition demonstration."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .implicit import implicit_tanh, init_contractive_params
from .inspection import jaxpr_text, stablehlo_text
from .safe_norm import safe_l2_norm


def main() -> None:
    params = init_contractive_params(jax.random.key(7), input_size=3, hidden_size=4)
    batch = jnp.array([[0.2, -0.1, 0.4], [0.1, 0.3, -0.2]], dtype=jnp.float32)

    batched_layer = jax.jit(jax.vmap(lambda row: implicit_tanh(params, row)))
    outputs = batched_layer(batch)

    def loss(model_params):
        values = jax.vmap(lambda row: implicit_tanh(model_params, row))(batch)
        return jnp.mean(values**2)

    loss_value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    gradient_norm = safe_l2_norm(
        jnp.concatenate([leaf.ravel() for leaf in jax.tree_util.tree_leaves(gradients)])
    )

    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print("Batched equilibrium output:")
    print(outputs)
    print(f"Loss: {float(loss_value):.8f}")
    print(f"Gradient L2 norm: {float(gradient_norm):.8f}")
    print("\nJAXPR excerpt:")
    print(jaxpr_text(lambda x: implicit_tanh(params, x), batch[0])[:1200])
    print("\nStableHLO excerpt:")
    print(stablehlo_text(lambda x: implicit_tanh(params, x), batch[0])[:1200])


if __name__ == "__main__":
    main()
