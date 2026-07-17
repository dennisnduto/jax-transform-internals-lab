# Publish this repository to GitHub

Create an empty GitHub repository named `jax-transform-internals-lab`, then run from this folder:

```bash
git init
git add .
git commit -m "Initial JAX internals portfolio implementation"
git branch -M main
git remote add origin https://github.com/dennisnduto/jax-transform-internals-lab.git
git push -u origin main
```

Before publishing, run `pytest -q`, read every source file, and be prepared to explain the
transformation, lowering, or sharding decisions in an interview.
