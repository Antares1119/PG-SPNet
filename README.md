# PG-SPNet source package

This package contains the PG-SPNet network implementation, HDF5 data reader, a 50-to-100 training entry point, the full-window evaluation code, and the fixed K=24 and K=9 masks. **Model checkpoints and CFD case files are available from the corresponding author upon reasonable request.** They are not included in this archive.

## Package and model structure

`model/` contains the pressure encoder, autoregressive pressure-dynamics predictor, force head, ODE integration, state correction, and loss computation. The model maps observed pressure and the latest observed state through pressure features and predicted forces to future structural response. `data/` contains the HDF5 reader and dataset specification. `train/` contains the shared trainer and the 50-to-100 full-sensor/fixed-mask entry point. `evaluate/` computes complete-window metrics. `layouts/` contains the two released fixed masks. The supplied source is the network and training implementation; the complete historical LOO ranking is not part of this archive.

## Data protocol

Input: 50 past pressure frames from 72 circumferential surface locations, together with the latest observed structural state. Output: 100 future response frames. The training loader skips the first 500 aligned steps and uses stride 5. The evaluation script selects windows whose forecast lies within 200–250 s at stride 5, using the held-out test cases. Metrics include pooled-window R2 and amplitude-normalized RMSE, plus per-case scores. See [the dataset contract](data/schema.md) for HDF5 groups, field shapes, time alignment, sensor ordering, and validation checks; see [the split instructions](data/re_split.md) for the Re case allocation.

## ODE settings

**Fill `--mass`, `--stiffness`, `--ode-damping`, `--dt-data`, and `--dt-solver` according to the actual structural model and data.** The source intentionally has no default values for these physical parameters. The optional learnable damping switch is `--ode-damping-learnable`; provide its initial damping value with `--ode-damping`. Do not infer physical parameters from the example commands.

## Training

Install dependencies with `pip install -r requirements.txt` from the package root. Provide the requested HDF5 files and use the following command structure. The `run.py` entry point can train with all 72 sensors from random initialization when `--mask-file` and `--resume` are omitted. It can warm-start a fixed-mask K=24 or K=9 model when they are supplied. Match architecture switches and hyperparameters to the experiment being reproduced.

```bash
python -m train.run --train-data /path/to/train.hdf5 --val-data /path/to/val.hdf5 --checkpoint-dir /path/to/output --mass M --stiffness K --ode-damping C --dt-data DT_DATA --dt-solver DT_SOLVER --ode-damping-learnable --decouple-xy-heads --y-residual-branch --ring-prior-enabled
```

For K=24 or K=9 add `--mask-file layouts/k24_mask.json` or `--mask-file layouts/k09_mask.json`, and `--resume /path/to/reference_checkpoint.pth`. The historical sparse adaptation used a maximum of 10 epochs; an individual run may converge and stop earlier. The released entry point is adapted from the fixed-mask training code. It provides a full-sensor random-initialization route, but does not claim to reconstruct an unavailable historical warm-start sequence exactly.

## Evaluation

```bash
python -m evaluate.full_window --test-data /path/to/test.hdf5 --checkpoint /path/to/k24_checkpoint.pth --mask-file layouts/k24_mask.json --output-dir /path/to/evaluation --mass M --stiffness K --ode-damping C --dt-data DT_DATA --dt-solver DT_SOLVER --dt DT_DATA --ode-damping-learnable --decouple-xy-heads --y-residual-branch --ring-prior-enabled
```

Use the K=9 mask for the K=9 checkpoint. Omit `--mask-file` for the full 72-sensor checkpoint. The code reports the exact case and pooled aggregates computed from the supplied checkpoint and data.

## Availability and limits

Code and the two fixed masks are included. Trained checkpoints and CFD case files can be requested from the corresponding author. Without those files, the reported trained-model results and layout rankings cannot be independently reproduced from this archive alone. The provided masks encode the selected layouts; this archive does not include a complete LOO ranking record.
