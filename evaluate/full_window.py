from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.viv_causal_predictor import CausalVIVPredictor

T_IN, T_OUT, STRIDE = 50, 100, 5

def load_mask(mask_file: Path | None) -> np.ndarray:
    if mask_file is None:
        return np.ones(72, dtype=np.float32)
    obj = json.loads(mask_file.read_text(encoding="utf-8"))
    arr = np.asarray(obj["mask"] if isinstance(obj, dict) and "mask" in obj else obj, dtype=np.float32)
    arr = (arr.reshape(-1) > 0.5).astype(np.float32)
    if arr.shape[0] != 72:
        raise ValueError(f"Mask length mismatch for {mask_file}: {arr.shape}")
    return arr

def prepare_state(pressure: np.ndarray, state_raw: np.ndarray) -> np.ndarray:
    if state_raw.shape[0] == pressure.shape[0]:
        return state_raw
    if state_raw.shape[0] == pressure.shape[0] * 2:
        return state_raw[::2]
    idx = np.linspace(0, state_raw.shape[0] - 1, pressure.shape[0]).round().astype(int)
    return state_raw[idx]

def infer_encoder_type(state_dict: dict[str, torch.Tensor]) -> str:
    return "x13_hybrid" if any(k.startswith("pressure_encoder_sparse.") for k in state_dict) else "x12"

def metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - gt
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((gt - gt.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    amp = float(0.5 * np.ptp(gt))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "nrmse_a": float(rmse / (amp + 1e-10)),
        "r2": float(1.0 - sse / (sst + 1e-10)),
        "sse": sse,
        "sst": sst,
        "amp_ref": amp,
    }

def make_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    cfg = ckpt.get("config", {})
    model = CausalVIVPredictor(
        num_pressure_sensors=72,
        pressure_hidden_dim=cfg.get("pressure_hidden_dim", 512),
        flow_hidden_dim=cfg.get("flow_hidden_dim", 512),
        pressure_temporal_layers=cfg.get("pressure_temporal_layers", 6),
        num_heads=cfg.get("num_heads", 8), dropout=0.0,
        num_harmonics=cfg.get("num_harmonics", 10),
        num_gru_layers=cfg.get("num_gru_layers", 2),
        pressure_encoder_type=cfg.get("pressure_encoder_type") or infer_encoder_type(state_dict),
        sensor_token_dim=cfg.get("sensor_token_dim", 96),
        sensor_spatial_layers=cfg.get("sensor_spatial_layers", 2),
        sensor_dropout_prob=0.0,
        context_encoder_type=cfg.get("context_encoder_type", "mean"),
        pressure_dynamics_type=cfg.get("pressure_dynamics_type", "enhanced"),
        num_attn_heads_pressure=cfg.get("num_attn_heads_pressure", 4),
        memory_len=cfg.get("memory_len", 50),
        decouple_xy_heads=args.decouple_xy_heads,
        y_residual_branch=args.y_residual_branch,
        ring_prior_enabled=args.ring_prior_enabled,
        ring_neighbor_alpha=args.ring_neighbor_alpha,
        ring_opposite_beta=args.ring_opposite_beta,
        mass=args.mass, stiffness=args.stiffness, damping=args.ode_damping,
        dt_data=args.dt_data, dt_solver=args.dt_solver,
        ode_damping_learnable=args.ode_damping_learnable,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    important_missing = [key for key in missing if "blackbox" not in key]
    if important_missing or unexpected:
        raise ValueError(f"Checkpoint architecture mismatch: missing={important_missing}, unexpected={unexpected}")
    return model.to(device).eval()

@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Evaluate complete 50-to-100 windows")
    p.add_argument("--test-data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mask-file")
    p.add_argument("--output-dir", required=True)
    for name in ("mass", "stiffness", "ode-damping", "dt-data", "dt-solver"):
        p.add_argument("--" + name, type=float, required=True, help="Fill according to the actual model")
    p.add_argument("--dt", type=float, required=True, help="Physical time step for selecting 200-250 s windows")
    p.add_argument("--start-time", type=float, default=200.0)
    p.add_argument("--end-time", type=float, default=250.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--decouple-xy-heads", action="store_true")
    p.add_argument("--y-residual-branch", action="store_true")
    p.add_argument("--ring-prior-enabled", action="store_true")
    p.add_argument("--ring-neighbor-alpha", type=float, default=0.10)
    p.add_argument("--ring-opposite-beta", type=float, default=0.06)
    p.add_argument("--ode-damping-learnable", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args, device)
    mask = load_mask(Path(args.mask_file) if args.mask_file else None)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    pools = {"x": [[], []], "y": [[], []]}
    with h5py.File(args.test_data, "r") as f:
        for key in sorted(f.keys(), key=lambda x: float(x)):
            grp = f[key]["cylinder"]
            pressure = grp["surface_pressure"][:].astype(np.float32)
            state = prepare_state(pressure, grp["state"][:].astype(np.float32))
            lo = max(0, round(args.start_time / args.dt) - T_IN)
            hi = min(len(pressure) - T_IN - T_OUT, round(args.end_time / args.dt) - T_IN - T_OUT)
            starts = list(range(lo, hi + 1, STRIDE))
            if not starts:
                raise ValueError(f"No evaluation windows for Re={key}; check time step and data length")
            case = {"x": [[], []], "y": [[], []]}
            for i in range(0, len(starts), args.batch_size):
                batch = starts[i:i + args.batch_size]
                p_in = np.stack([pressure[s:s + T_IN] for s in batch])
                init = np.stack([state[s + T_IN - 1] for s in batch])
                truth = np.stack([state[s + T_IN:s + T_IN + T_OUT] for s in batch])
                mask_tensor = torch.from_numpy(mask).to(device).unsqueeze(0).expand(len(batch), -1)
                pred = model(
                    surface_pressure_input=torch.from_numpy(p_in).to(device),
                    reynolds_number=torch.full((len(batch),), float(key), device=device),
                    initial_state=torch.from_numpy(init).to(device), T_out=T_OUT,
                    teacher_forcing_ratio=0.0, pressure_tf_ratio=0.0,
                    sensor_mask=mask_tensor,
                )["states"].cpu().numpy()
                for axis, index in (("x", 0), ("y", 1)):
                    case[axis][0].append(truth[:, :, index].reshape(-1))
                    case[axis][1].append(pred[:, :, index].reshape(-1))
            for axis in ("x", "y"):
                gt, pred = [np.concatenate(v) for v in case[axis]]
                rows.append({"re": float(key), "axis": axis, "windows": len(starts), **metrics(gt, pred)})
                pools[axis][0].append(gt)
                pools[axis][1].append(pred)
    pooled = {axis: metrics(np.concatenate(pools[axis][0]), np.concatenate(pools[axis][1])) for axis in ("x", "y")}
    case_mean = {axis: {name: float(np.mean([r[name] for r in rows if r["axis"] == axis])) for name in ("r2", "nrmse_a")} for axis in ("x", "y")}
    (outdir / "metrics.json").write_text(json.dumps({"pooled_window": pooled, "case_level_mean": case_mean, "per_case": rows}, indent=2), encoding="utf-8")
    with (outdir / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"pooled_window": pooled, "case_level_mean": case_mean}, indent=2))

if __name__ == "__main__":
    main()
