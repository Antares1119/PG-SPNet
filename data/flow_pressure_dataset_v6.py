import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict

class SurfacePressureDataset(Dataset):

    def __init__(
        self,
        hdf5_path: str,
        input_frames: int = 50,
        output_frames: int = 100,
        stride: int = 5,
        normalize: bool = True,
        anchor_x: bool = False,
        skip_head: int = 0,
    ):
        self.hdf5_path = hdf5_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.stride = stride
        self.normalize = normalize
        self.anchor_x = anchor_x   # ← subtract input-window x mean from all x
        self.skip_head = skip_head  # ← skip first N timesteps (remove drift transient)
        self.total_frames = input_frames + output_frames

        self._load_metadata()

    def _load_metadata(self):
        self.samples = []
        self.re_list = []

        with h5py.File(self.hdf5_path, 'r') as f:
            for re_key in sorted(f.keys(), key=lambda x: float(x)):
                re_value = float(re_key)

                total_timesteps = f[re_key]['cylinder']['force'].shape[0]

                for start in range(self.skip_head, total_timesteps - self.total_frames + 1, self.stride):
                    self.samples.append((re_value, start))

                self.re_list.append(re_value)

        self.re_list = sorted(set(self.re_list))
        print(f"  Dataset: {len(self.samples)} samples from {len(self.re_list)} Re values")
        print(f"  Re range: [{min(self.re_list):.0f}, {max(self.re_list):.0f}]")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        re, start_idx = self.samples[idx]
        end_idx = start_idx + self.total_frames

        with h5py.File(self.hdf5_path, 'r') as f:

            re_key = str(int(re)) if re == int(re) else str(re)
            grp = f[re_key]['cylinder']

            surface_pressure = grp['surface_pressure'][start_idx:end_idx]  # [T_total, 72]
            surface_pressure = torch.from_numpy(surface_pressure.astype(np.float32))

            force = grp['force'][start_idx:end_idx]         # [T_total, 2]
            force = torch.from_numpy(force.astype(np.float32))

            state_start = start_idx * 2
            state_end = end_idx * 2
            state_raw = grp['state'][state_start:state_end]  # [T_total*2, 4]
            state = state_raw[::2]
            state = torch.from_numpy(state.astype(np.float32))

        if self.normalize:

            pass

        T_in = self.input_frames

        if self.anchor_x:
            x_offset = state[:T_in, 0].mean()
            state = state.clone()
            state[:, 0] = state[:, 0] - x_offset
        else:
            x_offset = torch.tensor(0.0, dtype=torch.float32)

        return {

            'surface_pressure_input': surface_pressure[:T_in],    # [T_in, 72]

            'surface_pressure_output': surface_pressure[T_in:],   # [T_out, 72]
            'force_output': force[T_in:],                         # [T_out, 2]
            'state_output': state[T_in:],                         # [T_out, 4]

            'force_input': force[:T_in],                          # [T_in, 2]
            'state_input': state[:T_in],                          # [T_in, 4]

            'initial_state': state[T_in - 1],
            'reynolds_number': torch.tensor(re, dtype=torch.float32),

            'x_offset': x_offset,                                  # scalar
        }

def create_pressure_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    input_frames: int = 50,
    output_frames: int = 50,
    stride: int = 5,
    anchor_x: bool = False,
    skip_head: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    print(f"\n (V6 — ):")
    print(f"  Train: {train_path}")
    train_dataset = SurfacePressureDataset(
        hdf5_path=train_path,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
        normalize=True,
        anchor_x=anchor_x,
        skip_head=skip_head,
    )

    print(f"  Valid: {val_path}")
    val_dataset = SurfacePressureDataset(
        hdf5_path=val_path,
        input_frames=input_frames,
        output_frames=output_frames,
        stride=stride,
        normalize=True,
        anchor_x=anchor_x,
        skip_head=skip_head,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metadata = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_re_values': train_dataset.re_list,
        'val_re_values': val_dataset.re_list,
    }

    print(f"\n✓ Dataloaders created:")
    print(f"  Train: {metadata['train_samples']} samples, {len(train_loader)} batches (B={batch_size})")
    print(f"  Val:   {metadata['val_samples']} samples, {len(val_loader)} batches")

    return train_loader, val_loader, metadata

if __name__ == '__main__':
    print("=" * 70)
    print('')
    print("=" * 70)

    train_loader, val_loader, metadata = create_pressure_dataloaders(
        train_path='data/cylinder_training_calulation.hdf5',
        val_path='data/cylinder_valid_calulation.hdf5',
        batch_size=4,
        num_workers=0,
        input_frames=50,
        output_frames=50,
        stride=5,
    )

    print('')
    batch = next(iter(train_loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={list(value.shape)}, "
                  f"range=[{value.min().item():.4f}, {value.max().item():.4f}]")

    print('')
