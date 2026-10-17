import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import json
import argparse

from model.viv_causal_predictor import CausalVIVPredictor, compute_causal_viv_loss
from data.flow_pressure_dataset_v6 import SurfacePressureDataset

class End2EndTrainer:

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(), lr=config['lr'],
            weight_decay=config['weight_decay'], betas=(0.9, 0.999),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'] - config['warmup_epochs'],
            eta_min=config['min_lr'],
        )

        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.base_lr = config['lr']
        self.warmup_epochs = config['warmup_epochs']
        self.loss_weights = dict(config['loss_weights'])
        self.grad_clip = config['grad_clip']

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.history = []
        self.nan_batch_count = 0
        self.nan_epoch_streak = 0
        self.grad_accum_steps = config.get('grad_accum_steps', 1)

    def get_tf_ratio(self, epoch):
        s, e, d = self.config['tf_start'], self.config['tf_end'], self.config['tf_decay_epochs']
        return max(e, s + (e - s) * (epoch / d))

    def get_p_tf_ratio(self, epoch):
        s, e, d = self.config['p_tf_start'], self.config['p_tf_end'], self.config['p_tf_decay_epochs']
        return max(e, s + (e - s) * (epoch / d))

    def warmup_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

    def prepare_batch(self, batch):
        return {k: batch[k].to(self.device) for k in
                ['surface_pressure_input', 'surface_pressure_output',
                 'reynolds_number', 'initial_state', 'force_output', 'state_output']}

    def train_step(self, batch, epoch, step_in_epoch):
        b = self.prepare_batch(batch)
        tf_ratio = self.get_tf_ratio(epoch)
        p_tf_ratio = self.get_p_tf_ratio(epoch)
        is_accum_step = (step_in_epoch % self.grad_accum_steps != 0)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred = self.model(
                surface_pressure_input=b['surface_pressure_input'],
                reynolds_number=b['reynolds_number'],
                initial_state=b['initial_state'],
                T_out=b['force_output'].shape[1],
                true_states=b['state_output'],
                teacher_forcing_ratio=tf_ratio,
                true_pressure_output=b['surface_pressure_output'],
                pressure_tf_ratio=p_tf_ratio,
            )
            losses = compute_causal_viv_loss(
                pred, b, self.device, self.loss_weights, model=self.model,
            )
            scaled_loss = losses['total'] / self.grad_accum_steps

        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            self.nan_batch_count += 1
            return {k: float('nan') for k in losses}

        if self.scaler:
            self.scaler.scale(scaled_loss).backward()
            if not is_accum_step:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.nan_batch_count += 1
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    return {k: float('nan') for k in losses}
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            scaled_loss.backward()
            if not is_accum_step:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.step()
                else:
                    self.nan_batch_count += 1
                    self.optimizer.zero_grad()
                    return {k: float('nan') for k in losses}
                self.optimizer.zero_grad()

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def val_step(self, batch):
        b = self.prepare_batch(batch)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred = self.model(
                surface_pressure_input=b['surface_pressure_input'],
                reynolds_number=b['reynolds_number'],
                initial_state=b['initial_state'],
                T_out=b['force_output'].shape[1],
            )
            losses = compute_causal_viv_loss(
                pred, b, self.device, self.loss_weights, model=self.model,
            )
        return {k: v.item() for k, v in losses.items()}

    def aggregate(self, loss_list):
        result = {}
        for k in loss_list[0]:
            vals = [l[k] for l in loss_list if not np.isnan(l[k])]
            result[k] = np.mean(vals) if vals else float('nan')
        return result

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        tf = self.get_tf_ratio(epoch)
        ptf = self.get_p_tf_ratio(epoch)
        pbar = tqdm(
            self.train_loader,
            desc=f'Train E{epoch} (sTF={tf:.2f} pTF={ptf:.2f})',
            leave=False,
        )
        for step, batch in enumerate(pbar, 1):
            sl = self.train_step(batch, epoch, step)
            losses.append(sl)
            pbar.set_postfix(
                loss=f"{sl['total']:.4f}", F=f"{sl['force']:.4f}",
                P=f"{sl['position']:.4f}",
                xSm=f"{sl.get('x_smooth', 0):.4f}",
            )
        # Flush residual gradients
        if len(losses) % self.grad_accum_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    self.optimizer.step()
            self.optimizer.zero_grad()
        return self.aggregate(losses)

    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        return self.aggregate(
            [self.val_step(b) for b in tqdm(self.val_loader, desc=f'Val E{epoch}', leave=False)]
        )

    def save_checkpoint(self, epoch, vl, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': vl['total'],
            'val_force': vl['force'],
            'val_position': vl['position'],
            'val_cumulative_pos': vl.get('cumulative_pos', 0),
            'val_spectral': vl.get('spectral', 0),
            'config': self.config,
            'model_type': 'CausalVIVPredictor_End2End_v6',
            'training_mode': 'end2end_v6',
            'pressure_dynamics_type': 'enhanced',
        }
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
        if epoch % self.config['save_every'] == 0:
            torch.save(ckpt, self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth')

    def train(self, epochs, start_epoch=0):
        print(f"\n{'='*80}")
        print(f" v6:  + PressureContextEncoder (Re=50~1M)")
        print(f"{'='*80}")
        print(f": {len(self.train_loader.dataset)}, "
              f": {len(self.val_loader.dataset)}")
        print(f": EnhancedPressureDynamics + PressureContextEncoder, "
              f"attn_heads={self.config.get('num_attn_heads_pressure', 4)}, "
              f"memory_len={self.config.get('memory_len', 50)}")
        print(f": T_in={self.config['input_frames']}, "
              f"T_out={self.config['output_frames']}")
        if start_epoch > 0:
            print(f"\n→  Epoch {start_epoch} ")
        print(f"\nTF Schedule:")
        print(f"  State TF:    {self.config['tf_start']}→{self.config['tf_end']} "
              f"over {self.config['tf_decay_epochs']} epochs")
        print(f"  Pressure TF: {self.config['p_tf_start']}→{self.config['p_tf_end']} "
              f"over {self.config['p_tf_decay_epochs']} epochs")
        print(f"  Aux P-feat:   {self.loss_weights['pressure_feat']}")
        print(f"  Grad Accum:  {self.grad_accum_steps} "
              f"(batch={self.config['batch_size']*self.grad_accum_steps})")
        print(f"  LR: {self.config['lr']}, Patience: {self.config['early_stop_patience']}")

        start = time.time()
        for epoch in range(start_epoch + 1, epochs + 1):
            if epoch <= self.warmup_epochs:
                self.warmup_lr(epoch)

            tl = self.train_epoch(epoch)
            vl = self.val_epoch(epoch)

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            train_is_nan = np.isnan(tl['total'])
            val_is_nan = np.isnan(vl['total'])

            if train_is_nan:
                self.nan_epoch_streak += 1
                if self.nan_epoch_streak >= 3:
                    best_ckpt = self.checkpoint_dir / 'best_model.pth'
                    if best_ckpt.exists():
                        print(f"  ⚠ {self.nan_epoch_streak}NaN epochs! "
                              f"best model (Epoch {self.best_epoch})")
                        ckpt = torch.load(best_ckpt, map_location=self.device,
                                          weights_only=False)
                        self.model.load_state_dict(ckpt['model_state_dict'])
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = pg['lr'] * 0.5
                        print(f"  LR {self.optimizer.param_groups[0]['lr']:.2e}")
                        self.nan_epoch_streak = 0
                self.no_improve_count += 1
            else:
                self.nan_epoch_streak = 0

            is_best = (not val_is_nan) and vl['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = vl['total']
                self.best_epoch = epoch
                self.no_improve_count = 0
            elif not train_is_nan:
                self.no_improve_count += 1

            elapsed = (time.time() - start) / 3600
            stf = self.get_tf_ratio(epoch)
            ptf = self.get_p_tf_ratio(epoch)

            nan_info = f" nan_batches={self.nan_batch_count}" if self.nan_batch_count > 0 else ""
            self.nan_batch_count = 0

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: {tl['total']:.4f} (F:{tl['force']:.4f} Fx:{tl.get('force_x',0):.4f} "
                f"P:{tl['position']:.4f} DC-x:{tl.get('dc_x',0):.4f} "
                f"xSm:{tl.get('x_smooth',0):.4f}) | "
                f"Val: {vl['total']:.4f} (F:{vl['force']:.4f} Fx:{vl.get('force_x',0):.4f} "
                f"P:{vl['position']:.4f} DC-x:{vl.get('dc_x',0):.4f} xSm:{vl.get('x_smooth',0):.4f}) | "
                f"LR:{lr:.2e} sTF:{stf:.2f} pTF:{ptf:.2f} | {elapsed:.2f}h"
                + (" ← BEST" if is_best else "") + nan_info
            )

            if not val_is_nan:
                self.save_checkpoint(epoch, vl, is_best)
            self.history.append({
                'epoch': epoch, 'train': tl, 'val': vl,
                'lr': lr, 'state_tf': stf, 'pressure_tf': ptf,
            })

            if self.no_improve_count >= self.config['early_stop_patience']:
                print(f"\n⚠ Early Stopping at Epoch {epoch} "
                      f"(no improvement for {self.config['early_stop_patience']} epochs)")
                break

        final_epoch = epoch
        total_h = (time.time() - start) / 3600
        print(f"\n! {final_epoch} epochs, {total_h:.2f}h")
        print(f"Best Epoch {self.best_epoch}, Val Loss {self.best_val_loss:.6f}")

        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

