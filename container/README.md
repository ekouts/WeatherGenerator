# WeatherGenerator container track (Transformer Engine spike)

Runs WeatherGenerator on top of a [CSCS Alps Extended Image](https://docs.cscs.ch/software/alps-extended-images/):
NGC PyTorch plus the Slingshot networking stack (libfabric, patched NCCL 2.29.2,
AWS OFI plugin). Purpose: get Transformer Engine (TE) pre-installed and
version-matched instead of building it from source in the uenv.
This track is experimental and separate from the main uenv+venv workflow.

Base image: `ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.02-py3-alps6`

| component | version |
|---|---|
| PyTorch | 2.11.0a0 (NVIDIA build — *not* the repo's pinned 2.9.1) |
| TransformerEngine | 2.12 |
| NCCL | 2.29.2 (the ≥2.29 line qualified on Slingshot) |
| CUDA / cuDNN | 13.1 / 9.17 |
| Python | 3.12 |

How the layering works: `container/Containerfile` creates `/opt/venv` with
`--system-site-packages` and runs `uv sync` **without the gpu extra**, so all
WeatherGenerator deps (incl. the `packages/` workspace members and the
anemoi-datasets fork) are installed while torch / flash-attn / TE resolve to
the NGC stack. The base dependency closure contains no torch or CUDA packages,
so nothing in the image gets clobbered.

## Build on Santis

All commands are run on Santis unless noted.

### 1. One-time podman setup

```bash
mkdir -p $HOME/.config/containers
cat > $HOME/.config/containers/storage.conf <<'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
```

Note `/dev/shm` is wiped when the job ends — the image must be imported
(step 4) inside the same allocation as the build.

### 2. Interactive node

```bash
srun -A ch17 -p normal -t 02:00:00 --pty bash
```

### 3. Build (from the repo root)

```bash
cd <path to WeatherGenerator checkout>
podman build -f container/Containerfile -t wg-te:26.02-alps6 .
```

Needs internet access from the compute node (pulls the base image from GHCR
and clones the anemoi-datasets fork).

### 4. Import to squashfs (same allocation as step 3!)

```bash
mkdir -p $SCRATCH/images
enroot import -x mount -o $SCRATCH/images/wg-te-26.02-alps6.sqsh podman://wg-te:26.02-alps6
```

### 5. Point the EDF at the image

Edit `container/wg-te.toml`: replace `<username>` in `image` (and in the
optional live-code mount, if you enable it).

## Smoke tests

Single GPU — stack sanity (torch, flash-attn, TE, weathergen all importable):

```bash
srun -A ch17 -p normal -t 15 --environment=$PWD/container/wg-te.toml \
  python -c "
import torch, flash_attn, transformer_engine, transformer_engine.pytorch as te
import weathergen, weathergen.model.attention
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('flash-attn', flash_attn.__version__, '| TE', transformer_engine.__version__)
"
```

Unit tests:

```bash
srun -A ch17 -p normal -t 30 --environment=$PWD/container/wg-te.toml \
  python -m pytest tests/ -x -q
```

Multi-node jobs additionally need `--mpi=pmix --network=disable_rdzv_get` on
`srun` (per the extended-images docs). NCCL/libfabric env vars are set by the
image entrypoint — do not copy the uenv exports from `weathergen_slurm.sh`.

## Known caveats

- **flash-attn**: WG imports `flash_attn` at module level. NGC images ship it,
  but the support matrix doesn't pin its version — if the smoke test fails on
  this import, install it in the image against the container torch
  (`uv pip install flash-attn --no-build-isolation`, slow) and report back.
- **torch 2.11 alpha**: NGC ships NVIDIA snapshots ahead of PyPI. Deprecation
  warnings or behavior drift vs 2.9.1 are findings of this spike, not bugs to
  fix in main.
- **Perf comparisons**: measure baseline (flash-attn path) and TE path both
  *inside this container* — comparing container numbers against uenv numbers
  measures the torch upgrade, not TE.
- **Slingshot upgrades**: rebuilding means bumping `BASE_IMAGE` to the newer
  `-alpsN` tag and re-running steps 3–4.
