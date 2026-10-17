import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PressureTemporalEncoder_A(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 256,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_fourier_features: int = 6,
        sensor_feat_dim: int = 32,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.sensor_feat_dim = sensor_feat_dim

        angles = torch.linspace(0, 2 * np.pi * (num_sensors - 1) / num_sensors, num_sensors)
        fourier_feats = []
        for k in range(1, num_fourier_features + 1):
            fourier_feats.append(torch.sin(k * angles))
            fourier_feats.append(torch.cos(k * angles))
        # [num_sensors, 2*K]
        pos_encoding = torch.stack(fourier_feats, dim=-1)
        self.register_buffer('sensor_pos_encoding', pos_encoding)

        pos_dim = 2 * num_fourier_features

        sensor_input_dim = 1 + pos_dim
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_input_dim, sensor_feat_dim * 2),
            nn.GELU(),
            nn.Linear(sensor_feat_dim * 2, sensor_feat_dim),
            nn.GELU(),
        )

        agg_input_dim = num_sensors * sensor_feat_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(agg_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

        print(f"  PressureEncoder_A: fourier_K={num_fourier_features}, "
              f"sensor_feat={sensor_feat_dim}, agg_input={agg_input_dim}→{hidden_dim}")

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pressure: [B, T, 72]
        Returns:
            features: [B, T, hidden_dim]
        """
        B, T, S = pressure.shape

        # pressure: [B, T, 72] → [B, T, 72, 1]
        p = pressure.unsqueeze(-1)
        # pos: [72, 2K] → [1, 1, 72, 2K]
        pos = self.sensor_pos_encoding.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        x = torch.cat([p, pos], dim=-1)

        x = self.sensor_mlp(x)

        x = x.reshape(B, T, -1)

        feat = self.frame_encoder(x)

        feat = feat + self.pos_embed[:, :T, :]
        feat = self.temporal_transformer(feat)

        return feat

class CircularConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad_size = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):

        x = F.pad(x, (self.pad_size, self.pad_size), mode='circular')
        return self.conv(x)

class PressureTemporalEncoder_B(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 256,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_channels: tuple = (64, 128, 256),
        kernel_sizes: tuple = (7, 5, 3),
    ):
        super().__init__()
        self.num_sensors = num_sensors

        conv_layers = []
        in_ch = 1
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layers.extend([
                CircularConv1d(in_ch, out_ch, kernel_size=ks),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.Dropout(dropout * 0.5))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        pool_dim = conv_channels[-1] * 2  # avg + max
        self.frame_encoder = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

        print(f"  PressureEncoder_B: conv_channels={conv_channels}, "
              f"kernels={kernel_sizes}, pool_dim={pool_dim}→{hidden_dim}")

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pressure: [B, T, 72]
        Returns:
            features: [B, T, hidden_dim]
        """
        B, T, S = pressure.shape

        x = pressure.reshape(B * T, 1, S)

        x = self.conv_net(x)

        avg_pool = x.mean(dim=-1)   # [B*T, C_last]
        max_pool = x.max(dim=-1)[0] # [B*T, C_last]
        x = torch.cat([avg_pool, max_pool], dim=-1)  # [B*T, 2*C_last]

        feat = self.frame_encoder(x)

        feat = feat.reshape(B, T, -1)

        feat = feat + self.pos_embed[:, :T, :]
        feat = self.temporal_transformer(feat)

        return feat

class PressureTemporalEncoder_C(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 512,
        num_temporal_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_harmonics: int = 10,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_harmonics = num_harmonics

        self.frame_encoder = nn.Sequential(
            nn.Linear(num_sensors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        angles = torch.linspace(0, 2 * np.pi * (num_sensors - 1) / num_sensors, num_sensors)
        basis_list = [torch.ones(num_sensors)]
        for k in range(1, num_harmonics + 1):
            basis_list.append(torch.cos(k * angles))  # c_k_cos
            basis_list.append(torch.sin(k * angles))  # c_k_sin

        fourier_basis = torch.stack(basis_list, dim=0) / np.sqrt(num_sensors)
        self.register_buffer('fourier_basis', fourier_basis)

        num_phys_feats = 1 + 2 * num_harmonics  # K=10 → 21

        self.physics_net = nn.Sequential(
            nn.Linear(num_phys_feats, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.physics_scale = nn.Parameter(torch.zeros(1))

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

        print(f"  PressureEncoder_C: K={num_harmonics}, phys_feats={num_phys_feats}, "
              f"physics_scale=0.0 (learnable)")

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        B, T, S = pressure.shape

        main_feat = self.frame_encoder(pressure)  # [B, T, hidden_dim]

        fourier_coeffs = torch.matmul(pressure, self.fourier_basis.T)  # [B, T, 1+2K]
        physics_feat = self.physics_net(fourier_coeffs)  # [B, T, hidden_dim]

        feat = main_feat + self.physics_scale * physics_feat

        feat = feat + self.pos_embed[:, :T, :]
        feat = self.temporal_transformer(feat)

        return feat

class PressureTemporalEncoder_D(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 512,
        num_temporal_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_channels: tuple = (64, 128, 256),
        kernel_sizes: tuple = (7, 5, 3),
        n_attn_heads: int = 2,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.n_attn_heads = n_attn_heads

        angles = torch.linspace(0, 2 * np.pi * (num_sensors - 1) / num_sensors, num_sensors)
        pos_channels = torch.stack([torch.cos(angles), torch.sin(angles)], dim=0)  # [2, 72]
        self.register_buffer('input_pos_channels', pos_channels)

        conv_layers = []
        in_ch = 3
        for i, (out_ch, ks) in enumerate(zip(conv_channels, kernel_sizes)):
            conv_layers.extend([
                CircularConv1d(in_ch, out_ch, kernel_size=ks),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            if i < len(conv_channels) - 1:
                conv_layers.append(nn.Dropout(dropout * 0.5))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*conv_layers)

        conv_out_dim = conv_channels[-1]  # 256

        self.sensor_pos_embed = nn.Parameter(torch.zeros(1, num_sensors, conv_out_dim))
        nn.init.trunc_normal_(self.sensor_pos_embed, std=0.02)

        self.attn_proj = nn.Linear(conv_out_dim, n_attn_heads)

        pool_out_dim = n_attn_heads * conv_out_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(pool_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

        print(f"  PressureEncoder_D: conv_ch={conv_channels}, kernels={kernel_sizes}, "
              f"attn_heads={n_attn_heads}, pool_out={pool_out_dim}->{hidden_dim}")

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        B, T, S = pressure.shape

        p = pressure.reshape(B * T, 1, S)                                      # [B*T, 1, 72]
        pos = self.input_pos_channels.unsqueeze(0).expand(B * T, -1, -1)       # [B*T, 2, 72]
        x = torch.cat([p, pos], dim=1)                                          # [B*T, 3, 72]

        x = self.conv_net(x)
        x = x.transpose(1, 2)                                                   # [B*T, 72, conv_out]

        x = x + self.sensor_pos_embed                                            # [B*T, 72, conv_out]

        attn_logits = self.attn_proj(x)                                          # [B*T, 72, n_heads]
        attn_weights = F.softmax(attn_logits, dim=1)                             # softmax over sensors
        # weighted sum: [B*T, n_heads, conv_out]
        pooled = torch.einsum('bsn,bsd->bnd', attn_weights, x)
        pooled = pooled.reshape(B * T, -1)                                       # [B*T, n_heads*conv_out]

        feat = self.frame_encoder(pooled)                                        # [B*T, hidden_dim]
        feat = feat.reshape(B, T, -1)                                            # [B, T, hidden_dim]

        feat = feat + self.pos_embed[:, :T, :]
        feat = self.temporal_transformer(feat)

        return feat

class PressureTemporalEncoder_X13(nn.Module):
    pass

class SparseModalReconstructor(nn.Module):
    """Refine sparse harmonic coefficients and reconstruct dense proxy pressure."""

    def __init__(self, num_coeffs: int, num_sensors: int, hidden_dim: int = 128):
        super().__init__()
        self.num_coeffs = num_coeffs
        self.num_sensors = num_sensors

        self.coeff_refine = nn.Sequential(
            nn.Linear(num_coeffs + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_coeffs),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(num_coeffs + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        coeffs_raw: torch.Tensor,
        sparse_basis: torch.Tensor,
        active_ratio: torch.Tensor,
    ) -> dict:
        inp = torch.cat([coeffs_raw, active_ratio], dim=-1)
        delta = 0.1 * torch.tanh(self.coeff_refine(inp))
        coeffs_refined = coeffs_raw + delta
        recon_dense = torch.einsum("btf,sf->bts", coeffs_refined, sparse_basis)
        confidence = self.conf_head(inp)
        return {
            "coeffs_raw": coeffs_raw,
            "coeffs_refined": coeffs_refined,
            "recon_dense": recon_dense,
            "confidence": confidence,
        }

class PressureTemporalEncoder_X13(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 512,
        num_temporal_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_harmonics: int = 10,
        sensor_token_dim: int = 96,
        num_spatial_layers: int = 2,
        sensor_dropout_prob: float = 0.0,
        ridge_lambda: float = 1e-3,
        ring_prior_enabled: bool = False,
        ring_neighbor_alpha: float = 0.10,
        ring_opposite_beta: float = 0.06,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_harmonics = num_harmonics
        self.sensor_dropout_prob = sensor_dropout_prob
        self.ridge_lambda = ridge_lambda
        self.ring_prior_enabled = ring_prior_enabled

        angles = torch.linspace(0, 2 * np.pi * (num_sensors - 1) / num_sensors, num_sensors)
        sensor_pos = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # [S,2]
        self.register_buffer('sensor_pos_encoding', sensor_pos)

        basis_list = [torch.ones(num_sensors)]
        for k in range(1, num_harmonics + 1):
            basis_list.append(torch.cos(k * angles))
            basis_list.append(torch.sin(k * angles))
        sparse_basis = torch.stack(basis_list, dim=-1) / np.sqrt(num_sensors)  # [S,1+2K]
        self.register_buffer('sparse_basis', sparse_basis)
        self.modal_reconstructor = SparseModalReconstructor(
            num_coeffs=sparse_basis.shape[-1],
            num_sensors=num_sensors,
            hidden_dim=max(64, sensor_token_dim),
        )

        self.sensor_token = nn.Sequential(
            nn.Linear(4, sensor_token_dim),
            nn.LayerNorm(sensor_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sensor_token_dim, sensor_token_dim),
            nn.LayerNorm(sensor_token_dim),
        )
        self.sensor_pos_proj = nn.Linear(2, sensor_token_dim, bias=False)

        # tokens <- tokens + a * (left+right)/2 + b * opposite
        self.ring_neighbor_alpha = nn.Parameter(torch.tensor([ring_neighbor_alpha], dtype=torch.float32))
        self.ring_opposite_beta = nn.Parameter(torch.tensor([ring_opposite_beta], dtype=torch.float32))

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=sensor_token_dim,
            nhead=max(1, min(num_heads, sensor_token_dim // 16)),
            dim_feedforward=sensor_token_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer,
            num_layers=num_spatial_layers,
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(sensor_token_dim, sensor_token_dim // 2),
            nn.GELU(),
            nn.Linear(sensor_token_dim // 2, 1),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(sensor_token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        num_phys_feats = 1 + 2 * num_harmonics
        self.physics_net = nn.Sequential(
            nn.Linear(num_phys_feats, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.physics_scale = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=num_temporal_layers,
        )

        print(
            f"  PressureEncoder_X13: token_dim={sensor_token_dim}, spatial_layers={num_spatial_layers}, "
            f"harmonics={num_harmonics}, sensor_dropout={sensor_dropout_prob}, "
            f"ring_prior={ring_prior_enabled}"
        )

    def _apply_ring_prior(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.ring_prior_enabled:
            return tokens
        # tokens: [N, S, D]
        left = torch.roll(tokens, shifts=1, dims=1)
        right = torch.roll(tokens, shifts=-1, dims=1)
        neigh = 0.5 * (left + right)
        opposite = torch.roll(tokens, shifts=tokens.shape[1] // 2, dims=1)
        return tokens + self.ring_neighbor_alpha.view(1, 1, 1) * neigh + self.ring_opposite_beta.view(1, 1, 1) * opposite

    def _expand_sensor_mask(self, sensor_mask: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        B, T, S = pressure.shape
        if sensor_mask is None:
            return torch.ones(B, T, S, device=pressure.device, dtype=pressure.dtype)
        if sensor_mask.dim() == 2:
            sensor_mask = sensor_mask.unsqueeze(1).expand(-1, T, -1)
        elif sensor_mask.dim() == 3 and sensor_mask.shape[1] == 1:
            sensor_mask = sensor_mask.expand(-1, T, -1)
        return sensor_mask.to(device=pressure.device, dtype=pressure.dtype)

    def _apply_sensor_dropout(self, sensor_mask: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.sensor_dropout_prob <= 0:
            return sensor_mask
        keep = (torch.rand_like(sensor_mask) > self.sensor_dropout_prob).to(sensor_mask.dtype)
        dropped = sensor_mask * keep
        too_sparse = dropped.sum(dim=-1, keepdim=True) < 4
        return torch.where(too_sparse, sensor_mask, dropped)

    def _estimate_sparse_harmonics(
        self,
        pressure: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, S = pressure.shape
        N = B * T
        out_dtype = pressure.dtype

        with torch.autocast(device_type=pressure.device.type, enabled=False):
            p_flat = pressure.reshape(N, S).float()
            m_flat = sensor_mask.reshape(N, S).float()

            # Sanitize: NaN/Inf in pressure → zero out before linear solve
            p_flat = torch.nan_to_num(p_flat, nan=0.0, posinf=0.0, neginf=0.0)
            p_flat = torch.clamp(p_flat, -1e4, 1e4)

            basis = self.sparse_basis.to(device=pressure.device, dtype=torch.float32)  # [S,F]

            ata = torch.einsum('sf,ns,sg->nfg', basis, m_flat, basis)
            rhs = torch.einsum('sf,ns,ns->nf', basis, m_flat, p_flat)

            eye = torch.eye(ata.shape[-1], device=pressure.device, dtype=torch.float32).unsqueeze(0)
            # Use larger ridge (1e-2) for better conditioning and CUDA stability
            ata = ata + max(self.ridge_lambda, 1e-2) * eye

            try:

                coeffs = torch.linalg.lstsq(ata, rhs.unsqueeze(-1)).solution.squeeze(-1)
            except RuntimeError:

                coeffs = torch.zeros((N, ata.shape[-1]), device=pressure.device, dtype=torch.float32)

            # Guard against any residual NaN/Inf from degenerate batches
            coeffs = torch.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)

        return coeffs.reshape(B, T, -1).to(dtype=out_dtype)

    def reconstruct_modal_from_input(
        self,
        pressure: torch.Tensor,
        sensor_mask: torch.Tensor = None,
        use_dropout: bool = False,
    ) -> dict:
        """Compute modal coefficients and dense reconstruction from sparse pressure input."""
        squeeze_time = (pressure.dim() == 2)
        if squeeze_time:
            pressure = pressure.unsqueeze(1)

        sensor_mask = self._expand_sensor_mask(sensor_mask, pressure)
        if use_dropout:
            sensor_mask = self._apply_sensor_dropout(sensor_mask)
        pressure_masked = pressure * sensor_mask

        coeffs_raw = self._estimate_sparse_harmonics(pressure_masked, sensor_mask)
        active_ratio = sensor_mask.mean(dim=-1, keepdim=True)
        modal = self.modal_reconstructor(
            coeffs_raw=coeffs_raw,
            sparse_basis=self.sparse_basis.to(device=pressure.device, dtype=pressure.dtype),
            active_ratio=active_ratio,
        )
        modal["active_ratio"] = active_ratio
        modal["sensor_mask"] = sensor_mask
        modal["pressure_masked"] = pressure_masked
        if squeeze_time:
            for k in ["coeffs_raw", "coeffs_refined", "recon_dense", "confidence", "active_ratio", "sensor_mask", "pressure_masked"]:
                modal[k] = modal[k].squeeze(1)
        return modal

    def encode_frame_features(
        self,
        pressure: torch.Tensor,
        sensor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        squeeze_time = (pressure.dim() == 2)
        if squeeze_time:
            pressure = pressure.unsqueeze(1)

        B, T, S = pressure.shape
        sensor_mask = self._expand_sensor_mask(sensor_mask, pressure)
        sensor_mask = self._apply_sensor_dropout(sensor_mask)
        pressure_masked = pressure * sensor_mask

        pos = self.sensor_pos_encoding.to(device=pressure.device, dtype=pressure.dtype)
        pos = pos.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        token_input = torch.cat([
            pressure_masked.unsqueeze(-1),
            pos,
            sensor_mask.unsqueeze(-1),
        ], dim=-1)
        tokens = self.sensor_token(token_input)
        tokens = tokens + self.sensor_pos_proj(pos)

        tokens = tokens.reshape(B * T, S, -1)
        tokens = self._apply_ring_prior(tokens)
        mask_flat = sensor_mask.reshape(B * T, S)
        key_padding_mask = mask_flat < 0.5
        spatial_feat = self.spatial_transformer(tokens, src_key_padding_mask=key_padding_mask)

        attn_logits = self.attn_pool(spatial_feat).squeeze(-1)
        attn_logits = attn_logits.masked_fill(key_padding_mask, -1e4)
        attn_weights = F.softmax(attn_logits, dim=-1)

        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            uniform = torch.full_like(attn_weights, 1.0 / attn_weights.shape[-1])
            attn_weights = torch.where(all_masked, uniform, attn_weights)

        pooled = torch.einsum('ns,nsd->nd', attn_weights, spatial_feat).reshape(B, T, -1)
        token_feat = self.token_proj(pooled)

        modal = self.reconstruct_modal_from_input(pressure_masked, sensor_mask=sensor_mask, use_dropout=False)
        coeffs = modal["coeffs_refined"]
        physics_feat = self.physics_net(coeffs)
        active_ratio = sensor_mask.mean(dim=-1, keepdim=True)
        fusion_input = torch.cat([token_feat, physics_feat, active_ratio], dim=-1)
        learned_gate = torch.sigmoid(self.fusion_gate(fusion_input))
        prior_gate = 0.15 + 0.70 * active_ratio
        mix_gate = torch.clamp(0.5 * learned_gate + 0.5 * prior_gate, 0.05, 0.95)

        scaled_physics = self.physics_scale * physics_feat
        feat = mix_gate * token_feat + (1.0 - mix_gate) * scaled_physics
        feat = self.fusion_norm(feat)

        return feat.squeeze(1) if squeeze_time else feat

    def forward(
        self,
        pressure: torch.Tensor,
        sensor_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        feat = self.encode_frame_features(pressure, sensor_mask=sensor_mask)
        squeeze_time = (feat.dim() == 2)
        if squeeze_time:
            feat = feat.unsqueeze(1)

        B, T, _ = feat.shape
        feat = feat + self.pos_embed[:, :T, :]
        feat = self.temporal_transformer(feat)
        return feat.squeeze(1) if squeeze_time else feat
