# neural-precond-schwinger — Neural Preconditioners for the Lattice Schwinger Model

Learned preconditioners for the Wilson–Dirac normal operator $D^\dagger D$ of the
two-flavor Schwinger model (2D U(1) lattice gauge theory), implemented in JAX.

Two network families map a gauge configuration $U$ to an auxiliary field
$\tilde U$ that defines the preconditioner
$M^{-1} = (\tilde D^\dagger \tilde D)^p$ built from the same Wilson stencil:

- **FNO** — a 2D Fourier neural operator (`src/model/FNO2d.py`)
- **FCN** — a fully convolutional network (`src/model/complexCNN.py`)

Both are trained unsupervised with the inverse-approximation loss
$\| M^{-1} A v - v \|$ over Gaussian probe vectors (`src/utils/losses.py`),
and evaluated by preconditioned CG on held-out configurations.

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # CPU / Apple Silicon
uv sync --extra cuda   # NVIDIA CUDA 12 hosts
```

## Data

Training expects U(1) gauge configurations as a NumPy array of link phases
$\theta$ with shape `(N, 2, L, L)`, stored at
`data/U1Configs/config.l{L}-N1600-b2.0-k{kappa}-unquenched.npy`
(links are recovered as $U = e^{i\theta}$).

Configurations can be generated with
[JulianSchwingerModel](https://github.com/ylin910095/JulianSchwingerModel)
(HMC for the two-flavor Schwinger model, Julia).

## Train

From the `jax_model/` directory:

```bash
# FNO, e.g. L=16, kappa=0.276, preconditioner power p=1
python -m scripts.precondFNO_L --L 16 --kappa 0.276 --power 1

# FCN (hydra-configured; see configs/config.yaml)
python -m scripts.precondCNN_L data.L=16 data.kappa=0.276
```

Logs and checkpoints (`best_model.eqx`, `last_model.eqx`) are written under
`./logs/<run_name>/`. The FCN script also logs to Weights & Biases
(project set in `configs/config.yaml`).

## Test

```bash
# materialize D^dag D matrices for held-out configs
python -m scripts.make_testing_matrices --DL 16 --kappa 0.276

# preconditioned CG comparison (NN vs even-odd)
python -m scripts.evenOdd_cg --ML 16 --DL 16 --kappa 0.276

# CG with scipy (unpreconditioned / NN / incomplete Cholesky)
python -m scripts.scipy_cg --precond nn_pc
```

Plotting helpers live in `scripts/plot.py` / `scripts/make_plot.py`.

## Layout

```
jax_model/
  configs/       hydra config for FCN training
  scripts/       training, testing, and plotting entry points
  src/model/     FNO2d, complex CNN
  src/utils/     Wilson-Dirac operator, losses, CG solvers, iChol, metrics
  tests/         model/regression tests (pytest)
```

## Citation

If you use this code, please cite the accompanying paper on neural
preconditioners for lattice Dirac systems.
