import jax.numpy as jnp

from jax_transform_internals.inspection import jaxpr_text, stablehlo_text


def test_ir_helpers_return_expected_text():
    def fn(x):
        return jnp.sin(x) + x * x

    example = jnp.ones((4,), dtype=jnp.float32)
    assert "sin" in jaxpr_text(fn, example)
    hlo = stablehlo_text(fn, example)
    assert "stablehlo" in hlo
    assert "sine" in hlo or "sin" in hlo
