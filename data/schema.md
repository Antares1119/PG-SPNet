# Dataset contract

The training, validation, and test sets are supplied as three separate HDF5 files. Each top-level group name is a numeric Reynolds number. The data reader expects this structure in every file:

```text
<Re>/
└── cylinder/
    ├── surface_pressure   [3000, 72]
    ├── force              [3000, 2]
    ├── state              [6000, 4]
    └── moment             [3000, 2]  (present in the source cases; unused by this package)
```

`surface_pressure[t, j]` is the pressure at aligned time step `t` and circumferential sensor index `j` (zero based). The 72 sensor positions are ordered around the cylinder at 5° intervals; keep that ordering when applying the JSON masks. `force[t]` contains `[Fx, Fy]`. `state` contains `[x, y, vx, vy]` at twice the sampling rate of pressure and force. The reader selects `state[::2]` before constructing samples, so aligned pressure, force, and state each have 3,000 steps. Values are converted to `float32`. The training path leaves pressure values unchanged (`normalize=True` currently performs no transform) and uses `anchor_x=False`.

For a sample starting at aligned index `s`, the training input is `surface_pressure[s:s+50]`; targets are pressure, force, and response state over `[s+50:s+150]`. The initial state is the aligned state at `s+49`. Training and validation skip the first 500 aligned steps and advance the starting index by 5. The model therefore receives pressure history and the latest observed structural state. It does not receive the future pressure or force at inference.

The evaluation script uses the supplied held-out test file. At the time step given by `--dt`, it selects starts whose **forecast interval** lies within 200–250 s, with stride 5. For 3,000 aligned steps and `--dt 0.1`, this gives 81 windows per test case. It reports pooled-window metrics and case-level metrics separately. Its amplitude-normalized RMSE is `RMSE / (0.5 × [max(y_true) − min(y_true)])` for the relevant aggregation.

## Validation before use

- Check each case's groups, array shapes, time and force units, and sensor ordering and orientation against the CFD files.
- Verify that the three HDF5 files contain disjoint Re groups. Run `python -m data.inspect_split --train TRAIN.hdf5 --validation VALID.hdf5 --test TEST.hdf5 --output re_split.json` to export the exact case lists.
- Set the structural mass, stiffness, initial damping, data time step, and ODE solver time step for the case being used.

The CFD case files are available from the corresponding author upon request.
