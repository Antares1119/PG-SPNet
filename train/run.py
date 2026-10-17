#!/usr/bin/env python3
"""
Short finetuning with a fixed sensor mask using X13-hybrid model.

- Uses a fixed explicit sensor mask for both train and validation forward passes.
- Warm-starts from an existing checkpoint (default: x13 curriculum best).
- Intended for quick adaptation after mask selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure the code directory is on sys.path for imports
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.flow_pressure_dataset_v6 import SurfacePressureDataset
from train.trainer import End2EndTrainer
from model.viv_causal_predictor import CausalVIVPredictor, compute_causal_viv_loss

def infer_encoder_type_from_state_dict(state_dict: Dict[str, torch.Tensor], fallback: str = "x12") -> str:
    if any(k.startswith("pressure_encoder_sparse.") for k in state_dict.keys()):
        return "x13_hybrid"
    return fallback

def build_model_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    decouple_xy_heads: bool,
    y_residual_branch: bool,
    ring_prior_enabled: bool,
    ring_neighbor_alpha: float,
    ring_opposite_beta: float,
    re_ode_correction_enabled: bool,
    re_ode_correction_scale: float,
    re_ode_correction_re_threshold: float,
    mass: float, stiffness: float, damping: float, dt_data: float, dt_solver: float,
) -> CausalVIVPredictor:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    state_dict = ckpt.get("model_state_dict", {})
    encoder_type = cfg.get("pressure_encoder_type")
    if encoder_type is None:
        encoder_type = infer_encoder_type_from_state_dict(state_dict, fallback="x12")

    model = CausalVIVPredictor(
        num_pressure_sensors=72,
        pressure_hidden_dim=cfg.get("pressure_hidden_dim", 512),
        flow_hidden_dim=cfg.get("flow_hidden_dim", 512),
        pressure_temporal_layers=cfg.get("pressure_temporal_layers", 6),
        num_heads=cfg.get("num_heads", 8),
        dropout=cfg.get("dropout", 0.1),
        num_harmonics=cfg.get("num_harmonics", 10),
        num_gru_layers=cfg.get("num_gru_layers", 2),
        pressure_encoder_type=encoder_type,
        sensor_token_dim=cfg.get("sensor_token_dim", 96),
        sensor_spatial_layers=cfg.get("sensor_spatial_layers", 2),
        sensor_dropout_prob=0.0,
        context_encoder_type=cfg.get("context_encoder_type", "mean"),
        pressure_dynamics_type=cfg.get("pressure_dynamics_type", "enhanced"),
        num_attn_heads_pressure=cfg.get("num_attn_heads_pressure", 4),
        memory_len=cfg.get("memory_len", 50),
        mass=mass, stiffness=stiffness, damping=damping,
        dt_data=dt_data, dt_solver=dt_solver,
        decouple_xy_heads=decouple_xy_heads,
        y_residual_branch=y_residual_branch,
        ring_prior_enabled=ring_prior_enabled,
        ring_neighbor_alpha=ring_neighbor_alpha,
        ring_opposite_beta=ring_opposite_beta,
        re_ode_correction_enabled=re_ode_correction_enabled,
        re_ode_correction_scale=re_ode_correction_scale,
        re_ode_correction_re_threshold=re_ode_correction_re_threshold,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_mask(mask_file: Path, num_sensors: int = 72) -> np.ndarray:
    obj = json.loads(mask_file.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "mask" in obj:
        arr = np.asarray(obj["mask"], dtype=np.float32)
    elif isinstance(obj, dict) and "indices" in obj:
        arr = np.zeros(num_sensors, dtype=np.float32)
        arr[np.asarray(obj["indices"], dtype=np.int64)] = 1.0
    else:
        arr = np.asarray(obj, dtype=np.float32)
    arr = arr.reshape(-1)
    if arr.shape[0] != num_sensors:
        raise ValueError(f"mask length mismatch: got {arr.shape[0]}, expect {num_sensors}")
    arr = (arr > 0.5).astype(np.float32)
    if int(arr.sum()) < 4:
        raise ValueError("mask is too sparse (<4 active)")
    return arr

class FixedMaskTrainer(End2EndTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        config,
        mask_vec: np.ndarray,
        teacher_model: Optional[CausalVIVPredictor] = None,
    ):
        super().__init__(model, train_loader, val_loader, device, config)
        self.mask_vec = torch.from_numpy(mask_vec).float().to(device)
        self.y_force_aux = float(config.get("y_force_aux", 0.0))
        self.y_pos_aux = float(config.get("y_pos_aux", 0.0))
        self.distill_enabled = bool(config.get("distill_enabled", False))
        self.lambda_feat_distill = float(config.get("lambda_feat_distill", 0.0))
        self.lambda_out_distill = float(config.get("lambda_out_distill", 0.0))
        self.lambda_modal = float(config.get("lambda_modal", 0.0))
        self.lambda_spectral = float(config.get("lambda_spectral", 0.0))
        self.rollout_consistency_enabled = bool(config.get("rollout_consistency_enabled", False))
        self.lambda_rollout = float(config.get("lambda_rollout", 0.0))
        self.multiscale_enabled = bool(config.get("multiscale_enabled", False))
        self.near_frames = int(config.get("near_frames", 60))
        self.far_frames = int(config.get("far_frames", 60))
        self.far_stride = max(1, int(config.get("far_stride", 2)))
        self.modal_recon_enabled = bool(config.get("modal_recon_enabled", False))
        self.modal_confidence_enabled = bool(config.get("modal_confidence_enabled", False))
        self.teacher_model = teacher_model
        self.learnable_mask = bool(config.get("learnable_mask", False))
        self.mask_l1 = float(config.get("mask_l1", 0.0))
        self.mask_budget_weight = float(config.get("mask_budget_weight", 0.0))
        self.target_k = float(config.get("target_k", float(mask_vec.sum())))

        if self.learnable_mask:
            init_hi = float(config.get("mask_init_logit_active", 2.0))
            init_lo = float(config.get("mask_init_logit_inactive", -2.0))
            init_logits = torch.where(
                self.mask_vec > 0.5,
                torch.full_like(self.mask_vec, init_hi),
                torch.full_like(self.mask_vec, init_lo),
            )
            self.mask_logits = torch.nn.Parameter(init_logits)
            self.mask_lr = float(config.get("mask_lr", config.get("lr", 1e-6)))
            self.optimizer.add_param_group({
                "params": [self.mask_logits],
                "lr": self.mask_lr,
                "weight_decay": 0.0,
            })
        else:
            self.mask_logits = None

        if self.modal_recon_enabled:
            print("[Info] modal_recon_enabled=True (stage-2: real modal coeff + dense reconstruction loss)")

    def _compose_multiscale_input(self, pressure_input: torch.Tensor) -> torch.Tensor:
        if not self.multiscale_enabled:
            return pressure_input
        t_len = pressure_input.shape[1]
        near = pressure_input[:, -min(self.near_frames, t_len):, :]
        remain = pressure_input[:, :max(0, t_len - near.shape[1]), :]
        if remain.shape[1] <= 0 or self.far_frames <= 0:
            return near

        remain_ds = remain[:, ::self.far_stride, :]
        if remain_ds.shape[1] <= 0:
            return near

        if remain_ds.shape[1] <= self.far_frames:
            far = remain_ds
        else:
            idx = torch.linspace(0, remain_ds.shape[1] - 1, steps=self.far_frames, device=remain_ds.device)
            idx = idx.round().long()
            far = remain_ds.index_select(1, idx)
        return torch.cat([far, near], dim=1)

    def _spectral_loss_1d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()
        p_fft = torch.fft.rfft(pred_f, dim=1)
        t_fft = torch.fft.rfft(target_f, dim=1)
        mag = torch.mean(torch.abs(torch.abs(p_fft) - torch.abs(t_fft)))
        phase = torch.mean(torch.abs(torch.angle(p_fft) - torch.angle(t_fft)))
        return mag + 0.1 * phase

    def _get_sparse_encoder(self, model_obj) -> Optional[torch.nn.Module]:
        getter = getattr(model_obj, "_get_sparse_encoder", None)
        if callable(getter):
            return getter()
        return None

    def _add_distill_losses(self, losses: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], sensor_mask: torch.Tensor, pressure_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        losses["feat_distill"] = zero
        losses["out_distill"] = zero
        losses["modal_recon"] = zero
        losses["modal_coeff"] = zero
        losses["spectral_distill"] = zero
        losses["rollout_consistency"] = zero

        if not self.distill_enabled or self.teacher_model is None:
            return losses

        with torch.no_grad():
            full_mask = torch.ones_like(sensor_mask)
            teacher_pred = self.teacher_model(
                surface_pressure_input=pressure_input,
                reynolds_number=b["reynolds_number"],
                initial_state=b["initial_state"],
                T_out=b["force_output"].shape[1],
                teacher_forcing_ratio=0.0,
                pressure_tf_ratio=0.0,
                sensor_mask=full_mask,
            )

        feat_distill = torch.nn.functional.mse_loss(
            pred["pred_pressure_features"], teacher_pred["pred_pressure_features"]
        )
        out_distill = torch.nn.functional.mse_loss(pred["states"], teacher_pred["states"]) + torch.nn.functional.mse_loss(
            pred["forces"], teacher_pred["forces"]
        )
        y_spec = self._spectral_loss_1d(pred["states"][:, :, 1], teacher_pred["states"][:, :, 1])
        modal_recon_loss = self._spectral_loss_1d(pred["forces"][:, :, 1], teacher_pred["forces"][:, :, 1])
        modal_coeff_loss = torch.tensor(0.0, device=self.device)

        if self.modal_recon_enabled:
            student_sparse = self._get_sparse_encoder(self.model)
            teacher_sparse = self._get_sparse_encoder(self.teacher_model)
            if (
                student_sparse is not None
                and teacher_sparse is not None
                and hasattr(student_sparse, "reconstruct_modal_from_input")
                and hasattr(teacher_sparse, "reconstruct_modal_from_input")
            ):
                student_modal = student_sparse.reconstruct_modal_from_input(
                    pressure=pressure_input,
                    sensor_mask=sensor_mask,
                    use_dropout=False,
                )
                teacher_modal = teacher_sparse.reconstruct_modal_from_input(
                    pressure=pressure_input,
                    sensor_mask=full_mask,
                    use_dropout=False,
                )

                coeff_err = (student_modal["coeffs_refined"] - teacher_modal["coeffs_refined"]) ** 2
                recon_err = torch.abs(student_modal["recon_dense"] - teacher_modal["recon_dense"])
                if self.modal_confidence_enabled and "confidence" in teacher_modal:
                    conf = teacher_modal["confidence"].detach().clamp(0.05, 1.0)
                    coeff_err = coeff_err * conf
                    recon_err = recon_err * conf

                modal_coeff_loss = coeff_err.mean()
                modal_recon_loss = recon_err.mean()

        rollout_cons = torch.mean(
            torch.abs(
                (pred["states"][:, 1:, :] - pred["states"][:, :-1, :])
                - (teacher_pred["states"][:, 1:, :] - teacher_pred["states"][:, :-1, :])
            )
        )

        losses["total"] = losses["total"] + self.lambda_feat_distill * feat_distill + self.lambda_out_distill * out_distill + self.lambda_spectral * y_spec + self.lambda_modal * (modal_recon_loss + modal_coeff_loss)
        if self.rollout_consistency_enabled:
            losses["total"] = losses["total"] + self.lambda_rollout * rollout_cons

        losses["feat_distill"] = feat_distill
        losses["out_distill"] = out_distill
        losses["modal_recon"] = modal_recon_loss
        losses["modal_coeff"] = modal_coeff_loss
        losses["spectral_distill"] = y_spec
        losses["rollout_consistency"] = rollout_cons
        return losses

    def get_mask_prob(self) -> torch.Tensor:
        if not self.learnable_mask or self.mask_logits is None:
            return self.mask_vec
        return torch.sigmoid(self.mask_logits)

    def get_topk_hard_mask(self, k: int) -> torch.Tensor:
        probs = self.get_mask_prob().detach()
        k = max(1, min(int(k), probs.numel()))
        _, idx = torch.topk(probs, k=k)
        hard = torch.zeros_like(probs)
        hard[idx] = 1.0
        return hard

    def _mask_regularization(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.learnable_mask or self.mask_logits is None:
            losses["mask_l1"] = torch.tensor(0.0, device=self.device)
            losses["mask_budget"] = torch.tensor(0.0, device=self.device)
            losses["mask_active"] = torch.tensor(float(self.mask_vec.sum().item()), device=self.device)
            return losses

        probs = torch.sigmoid(self.mask_logits)
        l1_term = probs.mean()
        active = probs.sum()
        budget_term = ((active - self.target_k) / max(self.target_k, 1.0)) ** 2
        losses["total"] = losses["total"] + self.mask_l1 * l1_term + self.mask_budget_weight * budget_term
        losses["mask_l1"] = l1_term
        losses["mask_budget"] = budget_term
        losses["mask_active"] = active
        return losses

    def _batch_mask(self, batch_size: int) -> torch.Tensor:
        mask = self.get_mask_prob()
        return mask.unsqueeze(0).expand(batch_size, -1)

    def train_step(self, batch, epoch, step_in_epoch):
        b = self.prepare_batch(batch)
        tf_ratio = self.get_tf_ratio(epoch)
        p_tf_ratio = self.get_p_tf_ratio(epoch)
        is_accum_step = (step_in_epoch % self.grad_accum_steps != 0)
        pressure_input = self._compose_multiscale_input(b["surface_pressure_input"])
        sensor_mask = self._batch_mask(pressure_input.shape[0])

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred = self.model(
                surface_pressure_input=pressure_input,
                reynolds_number=b["reynolds_number"],
                initial_state=b["initial_state"],
                T_out=b["force_output"].shape[1],
                true_states=b["state_output"],
                teacher_forcing_ratio=tf_ratio,
                true_pressure_output=b["surface_pressure_output"],
                pressure_tf_ratio=p_tf_ratio,
                sensor_mask=sensor_mask,
                distill_alignment=self.distill_enabled,
            )
            losses = compute_causal_viv_loss(pred, b, self.device, self.loss_weights, model=self.model)
            if self.y_force_aux > 0:
                y_force_mae = torch.abs(pred["forces"][:, :, 1] - b["force_output"][:, :, 1]).mean()
                losses["total"] = losses["total"] + self.y_force_aux * y_force_mae
                losses["y_force_aux"] = y_force_mae
            else:
                losses["y_force_aux"] = torch.tensor(0.0, device=self.device)

            if self.y_pos_aux > 0:
                y_pos_mae = torch.abs(pred["states"][:, :, 1] - b["state_output"][:, :, 1]).mean()
                losses["total"] = losses["total"] + self.y_pos_aux * y_pos_mae
                losses["y_pos_aux"] = y_pos_mae
            else:
                losses["y_pos_aux"] = torch.tensor(0.0, device=self.device)
            losses = self._add_distill_losses(losses, pred, b, sensor_mask, pressure_input)
            losses = self._mask_regularization(losses)

            scaled_loss = losses["total"] / self.grad_accum_steps

        if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
            self.nan_batch_count += 1
            return {k: float("nan") for k in losses}

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
            if not is_accum_step:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.nan_batch_count += 1
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    return {k: float("nan") for k in losses}
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            scaled_loss.backward()
            if not is_accum_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.step()
                else:
                    self.nan_batch_count += 1
                    self.optimizer.zero_grad()
                    return {k: float("nan") for k in losses}
                self.optimizer.zero_grad()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def val_step(self, batch):
        b = self.prepare_batch(batch)
        pressure_input = self._compose_multiscale_input(b["surface_pressure_input"])
        sensor_mask = self._batch_mask(pressure_input.shape[0])
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred = self.model(
                surface_pressure_input=pressure_input,
                reynolds_number=b["reynolds_number"],
                initial_state=b["initial_state"],
                T_out=b["force_output"].shape[1],
                sensor_mask=sensor_mask,
                distill_alignment=self.distill_enabled,
            )
            losses = compute_causal_viv_loss(pred, b, self.device, self.loss_weights, model=self.model)
            if self.y_force_aux > 0:
                y_force_mae = torch.abs(pred["forces"][:, :, 1] - b["force_output"][:, :, 1]).mean()
                losses["total"] = losses["total"] + self.y_force_aux * y_force_mae
                losses["y_force_aux"] = y_force_mae
            else:
                losses["y_force_aux"] = torch.tensor(0.0, device=self.device)

            if self.y_pos_aux > 0:
                y_pos_mae = torch.abs(pred["states"][:, :, 1] - b["state_output"][:, :, 1]).mean()
                losses["total"] = losses["total"] + self.y_pos_aux * y_pos_mae
                losses["y_pos_aux"] = y_pos_mae
            else:
                losses["y_pos_aux"] = torch.tensor(0.0, device=self.device)
            losses = self._add_distill_losses(losses, pred, b, sensor_mask, pressure_input)
            losses = self._mask_regularization(losses)
        return {k: v.item() for k, v in losses.items()}

def main() -> None:
    p = argparse.ArgumentParser(description="Short fixed-mask finetune for X13 model")
    p.add_argument("--resume", default=None, help="Optional warm-start checkpoint")
    p.add_argument("--mask-file", default=None, help="Optional 72-element fixed mask JSON; omit for all sensors")
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--teacher-ckpt", default=None, help="Teacher checkpoint for distillation")
    p.add_argument("--distill-enabled", action="store_true", help="Enable teacher-student distillation")
    p.add_argument("--lambda-feat-distill", type=float, default=0.0)
    p.add_argument("--lambda-out-distill", type=float, default=0.0)
    p.add_argument("--lambda-modal", type=float, default=0.0)
    p.add_argument("--lambda-spectral", type=float, default=0.0)
    p.add_argument("--rollout-consistency-enabled", action="store_true")
    p.add_argument("--lambda-rollout", type=float, default=0.0)
    p.add_argument("--input-frames", type=int, default=50)
    p.add_argument("--num-harmonics", type=int, default=10)
    p.add_argument("--multiscale-enabled", action="store_true")
    p.add_argument("--near-frames", type=int, default=60)
    p.add_argument("--far-frames", type=int, default=60)
    p.add_argument("--far-stride", type=int, default=2)
    p.add_argument("--modal-recon-enabled", action="store_true", help="Stage-1 placeholder; enables modal proxy losses")
    p.add_argument("--modal-confidence-enabled", action="store_true", help="Stage-1 placeholder")
    p.add_argument("--y-force-aux", type=float, default=0.0, help="Aux weight on y-force MAE")
    p.add_argument("--y-pos-aux", type=float, default=0.0, help="Aux weight on y-position MAE")
    p.add_argument("--decouple-xy-heads", action="store_true", help="Enable XY-decoupled refinement heads")
    p.add_argument("--y-residual-branch", action="store_true", help="Enable lightweight y-only residual branch")
    p.add_argument("--ring-prior-enabled", action="store_true", help="Enable explicit circular prior in X13 encoder")
    p.add_argument("--ring-neighbor-alpha", type=float, default=0.10)
    p.add_argument("--ring-opposite-beta", type=float, default=0.06)
    p.add_argument("--re-ode-correction-enabled", action="store_true", help="Enable bounded Re-adaptive ODE correction")
    p.add_argument("--re-ode-correction-scale", type=float, default=0.0, help="Scale for bounded Re-adaptive ODE correction")
    p.add_argument("--re-ode-correction-re-threshold", type=float, default=3e5, help="Hard Re threshold to enable Re-ODE correction")
    p.add_argument("--no-ode-scaffold", action="store_true", help="Disable ODE scaffold (V1 experiment)")
    p.add_argument("--use-blackbox-state", action="store_true", help="Use black-box state updater instead of ODE (V5 experiment)")
    p.add_argument("--ode-damping", type=float, required=True, help="Set according to the actual structural model")
    p.add_argument("--mass", type=float, required=True, help="Set according to the actual structural model")
    p.add_argument("--stiffness", type=float, required=True, help="Set according to the actual structural model")
    p.add_argument("--dt-data", type=float, required=True, help="Set according to the actual data")
    p.add_argument("--dt-solver", type=float, required=True, help="Set according to the actual solver")
    p.add_argument("--ode-damping-learnable", action="store_true", help="Make ODE damping learnable (V4 experiment)")
    p.add_argument("--learnable-mask", action="store_true", help="Enable learnable soft sensor gating")
    p.add_argument("--target-k", type=int, default=36, help="Target active sensor count for learnable mask")
    p.add_argument("--mask-lr", type=float, default=1e-6, help="Learning rate for mask logits")
    p.add_argument("--mask-l1", type=float, default=5e-4, help="L1 weight on soft mask activity")
    p.add_argument("--mask-budget-weight", type=float, default=2e-2, help="Budget penalty weight")
    p.add_argument(
        "--loss-profile",
        choices=["default", "a"],
        default="default",
        help="Loss weight preset. 'a' emphasizes x-channel DC/AC and peak envelope.",
    )
    p.add_argument("--dc-x-w", type=float, default=None, help="Override loss_weights.dc_x")
    p.add_argument("--ac-x-w", type=float, default=None, help="Override loss_weights.ac_x")
    p.add_argument("--x-pearson-w", type=float, default=None, help="Override loss_weights.x_pearson")
    p.add_argument("--x-peak-w", type=float, default=None, help="Override loss_weights.x_peak")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    resume = Path(args.resume).resolve() if args.resume else None
    teacher_ckpt = (script_dir / args.teacher_ckpt).resolve() if args.teacher_ckpt else None
    mask_file = Path(args.mask_file).resolve() if args.mask_file else None
    train_data = Path(args.train_data).resolve()
    val_data = Path(args.val_data).resolve()

    device = torch.device(args.device)
    mask_vec = load_mask(mask_file, num_sensors=72) if mask_file else np.ones(72, dtype=np.float32)

    cfg = {
        "pressure_hidden_dim": 512,
        "flow_hidden_dim": 512,
        "pressure_temporal_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "num_harmonics": args.num_harmonics,
        "num_gru_layers": 2,
        "pressure_dynamics_type": "enhanced",
        "num_attn_heads_pressure": 4,
        "memory_len": 50,
        "input_frames": args.input_frames,
        "output_frames": 100,
        "stride": 5,
        "batch_size": args.batch_size,
        "grad_accum_steps": 1,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "min_lr": 1e-7,
        "grad_clip": 0.5,
        "tf_start": 0.0,
        "tf_end": 0.0,
        "tf_decay_epochs": 1,
        "p_tf_start": 0.0,
        "p_tf_end": 0.0,
        "p_tf_decay_epochs": 1,
        "loss_weights": {
            "force": 1.0,
            "position": 5.0,
            "velocity": 2.0,
            "variance": 0.2,
            "smoothness": 0.02,
            "pressure_feat": 0.0,
            "cumulative_pos": 0.8,
            "spectral": 0.3,
            "temporal_alpha": 1.0,
            "dc_x": 2.0,
            "ac_x": 1.5,
            "x_smooth": 0.0,
            "norm_pos": 0.2,
            "norm_vel": 0.2,
            "x_pearson": 0.5,
            "norm_force_x": 3.0,
            "norm_pos_adaptive": 2.0,
        },
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "save_every": 2,
        "early_stop_patience": 8,
        "y_force_aux": args.y_force_aux,
        "y_pos_aux": args.y_pos_aux,
        "learnable_mask": args.learnable_mask,
        "target_k": args.target_k,
        "mask_lr": args.mask_lr,
        "mask_l1": args.mask_l1,
        "mask_budget_weight": args.mask_budget_weight,
        "mask_init_logit_active": 2.0,
        "mask_init_logit_inactive": -2.0,
        "distill_enabled": args.distill_enabled,
        "lambda_feat_distill": args.lambda_feat_distill,
        "lambda_out_distill": args.lambda_out_distill,
        "lambda_modal": args.lambda_modal,
        "lambda_spectral": args.lambda_spectral,
        "rollout_consistency_enabled": args.rollout_consistency_enabled,
        "lambda_rollout": args.lambda_rollout,
        "multiscale_enabled": args.multiscale_enabled,
        "near_frames": args.near_frames,
        "far_frames": args.far_frames,
        "far_stride": args.far_stride,
        "modal_recon_enabled": args.modal_recon_enabled,
        "modal_confidence_enabled": args.modal_confidence_enabled,
    }

    if args.loss_profile == "a":
        cfg["loss_weights"].update({
            "dc_x": 3.0,
            "ac_x": 2.0,
            "x_pearson": 0.8,
            "x_peak": 1.5,
        })

    # Optional lightweight profile tuning from CLI (used for quick A-lite screening).
    if args.dc_x_w is not None:
        cfg["loss_weights"]["dc_x"] = float(args.dc_x_w)
    if args.ac_x_w is not None:
        cfg["loss_weights"]["ac_x"] = float(args.ac_x_w)
    if args.x_pearson_w is not None:
        cfg["loss_weights"]["x_pearson"] = float(args.x_pearson_w)
    if args.x_peak_w is not None:
        cfg["loss_weights"]["x_peak"] = float(args.x_peak_w)

    train_dataset = SurfacePressureDataset(
        str(train_data),
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride"],
        normalize=True,
        anchor_x=False,
        skip_head=500,
    )
    val_dataset = SurfacePressureDataset(
        str(val_data),
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride"],
        normalize=True,
        anchor_x=False,
        skip_head=500,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = CausalVIVPredictor(
        num_pressure_sensors=72,
        pressure_hidden_dim=cfg["pressure_hidden_dim"],
        flow_hidden_dim=cfg["flow_hidden_dim"],
        pressure_temporal_layers=cfg["pressure_temporal_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        num_harmonics=cfg["num_harmonics"],
        num_gru_layers=cfg["num_gru_layers"],
        mass=args.mass,
        stiffness=args.stiffness,
        damping=args.ode_damping,
        dt_data=args.dt_data,
        dt_solver=args.dt_solver,
        pressure_encoder_type="x13_hybrid",
        sensor_token_dim=96,
        sensor_spatial_layers=2,
        sensor_dropout_prob=0.0,
        context_encoder_type="mean",
        pressure_dynamics_type=cfg["pressure_dynamics_type"],
        num_attn_heads_pressure=cfg["num_attn_heads_pressure"],
        memory_len=cfg["memory_len"],
        decouple_xy_heads=args.decouple_xy_heads,
        y_residual_branch=args.y_residual_branch,
        ring_prior_enabled=args.ring_prior_enabled,
        ring_neighbor_alpha=args.ring_neighbor_alpha,
        ring_opposite_beta=args.ring_opposite_beta,
        re_ode_correction_enabled=args.re_ode_correction_enabled,
        re_ode_correction_scale=args.re_ode_correction_scale,
        re_ode_correction_re_threshold=args.re_ode_correction_re_threshold,
        no_ode_scaffold=args.no_ode_scaffold,
        use_blackbox_state=args.use_blackbox_state,
        ode_damping_learnable=args.ode_damping_learnable,
    ).to(device)

    if resume is not None:
        ckpt = torch.load(str(resume), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    teacher_model = None
    if args.distill_enabled and teacher_ckpt is not None:
        teacher_model = build_model_from_ckpt(
            ckpt_path=teacher_ckpt,
            device=device,
            decouple_xy_heads=args.decouple_xy_heads,
            y_residual_branch=args.y_residual_branch,
            ring_prior_enabled=args.ring_prior_enabled,
            ring_neighbor_alpha=args.ring_neighbor_alpha,
            ring_opposite_beta=args.ring_opposite_beta,
            re_ode_correction_enabled=args.re_ode_correction_enabled,
            re_ode_correction_scale=args.re_ode_correction_scale,
            re_ode_correction_re_threshold=args.re_ode_correction_re_threshold,
            mass=args.mass, stiffness=args.stiffness, damping=args.ode_damping,
            dt_data=args.dt_data, dt_solver=args.dt_solver,
        ).eval()
        for p_ in teacher_model.parameters():
            p_.requires_grad = False
        print(f"Teacher loaded and frozen: {teacher_ckpt}")

    print(f"Warm-start checkpoint: {resume if resume else 'none (random initialization)'}")
    print(f"Mask active sensors: {int(mask_vec.sum())}/72")
    print(f"Y-focus aux weights: y_force_aux={args.y_force_aux}, y_pos_aux={args.y_pos_aux}")
    print(f"Loss profile: {args.loss_profile}")
    print(
        "Loss overrides: "
        f"dc_x={cfg['loss_weights'].get('dc_x', 0)}, "
        f"ac_x={cfg['loss_weights'].get('ac_x', 0)}, "
        f"x_pearson={cfg['loss_weights'].get('x_pearson', 0)}, "
        f"x_peak={cfg['loss_weights'].get('x_peak', 0)}"
    )
    print(f"Decoupled XY heads: {args.decouple_xy_heads}")
    print(
        f"Y residual branch: {args.y_residual_branch} | "
        f"Ring prior: {args.ring_prior_enabled} "
        f"(neighbor={args.ring_neighbor_alpha}, opposite={args.ring_opposite_beta})"
    )
    print(
        f"Re-ODE correction: {args.re_ode_correction_enabled} "
        f"(scale={args.re_ode_correction_scale}, re_threshold={args.re_ode_correction_re_threshold})"
    )
    print(
        f"Distill: {args.distill_enabled}, teacher={teacher_ckpt if teacher_ckpt else 'None'} | "
        f"lambda_feat={args.lambda_feat_distill}, lambda_out={args.lambda_out_distill}, "
        f"lambda_modal={args.lambda_modal}, lambda_spectral={args.lambda_spectral}, "
        f"rollout={args.rollout_consistency_enabled}:{args.lambda_rollout}"
    )
    print(
        f"Input frames: {args.input_frames} | multiscale={args.multiscale_enabled} "
        f"(near={args.near_frames}, far={args.far_frames}, stride={args.far_stride})"
    )
    print(
        f"Modal recon flags: enabled={args.modal_recon_enabled}, confidence={args.modal_confidence_enabled}"
    )

    trainer = FixedMaskTrainer(
        model,
        train_loader,
        val_loader,
        device,
        cfg,
        mask_vec=mask_vec,
        teacher_model=teacher_model,
    )
    trainer.train(cfg["epochs"], start_epoch=0)

    if args.learnable_mask:
        ckpt_dir = Path(cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        prob = trainer.get_mask_prob().detach().cpu().numpy().astype(np.float32)
        topk = trainer.get_topk_hard_mask(args.target_k).detach().cpu().numpy().astype(np.float32)
        topk_idx = np.where(topk > 0.5)[0].astype(int).tolist()

        soft_json = ckpt_dir / "learned_soft_mask.json"
        hard_json = ckpt_dir / f"learned_top{int(args.target_k)}_mask.json"

        soft_payload = {
            "type": "soft_mask",
            "target_k": int(args.target_k),
            "active_sum": float(prob.sum()),
            "mask_prob": prob.tolist(),
        }
        hard_payload = {
            "type": "hard_mask_topk",
            "target_k": int(args.target_k),
            "indices": topk_idx,
            "keep_angles_deg": [int(i * 5) for i in topk_idx],
            "mask": topk.tolist(),
        }
        soft_json.write_text(json.dumps(soft_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        hard_json.write_text(json.dumps(hard_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved learned soft mask: {soft_json}")
        print(f"Saved learned hard mask: {hard_json}")

if __name__ == "__main__":
    main()
