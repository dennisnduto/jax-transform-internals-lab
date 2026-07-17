# JAX Transform Internals Lab

A focused, test-driven portfolio repository for JAX automatic-differentiation internals. It
contains two non-toy examples:

1. **`safe_l2_norm`** uses `jax.custom_jvp` to define a finite derivative convention at the
   non-differentiable zero vector while remaining compatible with `jit`, `jvp`, `grad`, and
   `vmap`.
2. **`implicit_tanh`** uses `jax.custom_vjp` to differentiate a converged equilibrium layer by
   solving the implicit adjoint system instead of backpropagating through every fixed-point
   iteration.

The repository also emits JAXPR and StableHLO text and tests nested transformation combinations.
It is a portfolio implementation, not an upstream JAX contribution.

## Why this is internals-level work

- Custom forward- and reverse-mode derivative rules.
- Residual design for a custom VJP.
- Matrix-free adjoint solve with `jax.scipy.sparse.linalg.gmres`.
- PyTree gradients for `(W, U, b)` parameters.
- Composition under `jit`, `vmap`, `grad`, `jvp`, and `value_and_grad`.
- Numerical finite-difference and unrolled-reference validation.
- JAXPR and StableHLO inspection.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python -m jax_transform_internals.demo
```

## Repository map

```text
src/jax_transform_internals/
  safe_norm.py       custom_jvp example
  implicit.py        custom_vjp equilibrium layer
  inspection.py      JAXPR and StableHLO helpers
  demo.py            reproducible end-to-end demonstration
tests/
  test_safe_norm.py
  test_implicit.py
  test_inspection.py
examples/
  compare_gradients.py
```

## Design notes

The equilibrium update is

```text
y* = tanh(W y* + U x + b)
```

For an output cotangent `g`, the backward rule solves

```text
(I - dF/dy)^T u = g
```

and then applies the VJP of `F` with respect to parameters and input using `u`. This separates
primal convergence from derivative propagation and avoids retaining the entire iteration tape.

## Scope and limitations

- The sample is intended for contractive systems; helper initialization scales `W` accordingly.
- `custom_vjp` supports reverse-mode differentiation but intentionally does not expose a
  forward-mode JVP for the equilibrium function.
- GMRES tolerances are examples and should be tuned for a production model.
- CPU tests validate correctness. GPU/TPU execution uses the same JAX program when an accelerator
  build of JAX is installed.
