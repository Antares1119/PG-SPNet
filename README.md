# PG-SPNet

This repository contains the PG-SPNet model, HDF5 data reader, training and evaluation scripts, and fixed K=24 and K=9 sensor masks for pressure-based prediction of vortex-induced vibration response.

## Package and model structure

`model/` contains the pressure encoder, autoregressive pressure predictor, force head, ODE solver, state correction, and loss functions. `data/` contains the HDF5 reader and data specification. `train/` contains the trainer and an entry point for full-sensor training or fixed-mask adaptation. `evaluate/` computes full-window metrics. `layouts/` contains the K=24 and K=9 masks.

## Data protocol

The model uses 50 past pressure frames from 72 circumferential sensor locations and the latest observed structural state to predict 100 future response frames. The training loader skips the first 500 aligned steps and uses stride 5. Evaluation uses windows whose forecast interval lies within 200–250 s and reports pooled-window and per-case metrics. See the [dataset specification](data/schema.md) for fields and time alignment, and the [Reynolds-number split](data/re_split.md) for the data allocation.

## ODE settings

Set `--mass`, `--stiffness`, `--ode-damping`, `--dt-data`, and `--dt-solver` for the structural model and sampling protocol being used. These arguments have no defaults. Use `--ode-damping-learnable` to optimize damping from the supplied initial value.

## Training

Install dependencies with `pip install -r requirements.txt` from the repository root. The command below trains with all 72 sensors from random initialization when `--mask-file` and `--resume` are omitted. Set the ODE arguments to values appropriate for the supplied data and model.

```bash
python -m train.run --train-data /path/to/train.hdf5 --val-data /path/to/val.hdf5 --checkpoint-dir /path/to/output --mass M --stiffness K --ode-damping C --dt-data DT_DATA --dt-solver DT_SOLVER --ode-damping-learnable --decouple-xy-heads --y-residual-branch --ring-prior-enabled
```

For fixed-mask adaptation, add `--mask-file layouts/k24_mask.json` or `--mask-file layouts/k09_mask.json` and `--resume /path/to/reference_checkpoint.pth`. Set `--epochs`, `--batch-size`, and other training options for the intended experiment.

## Evaluation

```bash
python -m evaluate.full_window --test-data /path/to/test.hdf5 --checkpoint /path/to/k24_checkpoint.pth --mask-file layouts/k24_mask.json --output-dir /path/to/evaluation --mass M --stiffness K --ode-damping C --dt-data DT_DATA --dt-solver DT_SOLVER --dt DT_DATA --ode-damping-learnable --decouple-xy-heads --y-residual-branch --ring-prior-enabled
```

Use the K=9 mask for a K=9 checkpoint. Omit `--mask-file` for a full 72-sensor checkpoint. Results are written as pooled-window and per-case metrics.

## Data and checkpoint availability

Trained model checkpoints and CFD case files are available from the corresponding author upon request.
