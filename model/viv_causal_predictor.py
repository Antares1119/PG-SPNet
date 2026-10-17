import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from model.viv_pressure_predictor import (
    PhysicsODESolver,
    FlowContextEncoder,
    PressureContextEncoder,
    AttentivePressureContextEncoder,
    ForceHead,
    StateCorrectionHead,
)
from model.spatial_pressure_encoders import PressureTemporalEncoder_C, PressureTemporalEncoder_X13

class PressureDynamicsPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 512,
        re_dim: int = 64,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        condition_dim = 4 + hidden_dim + re_dim
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        print(f"  PressureDynamicsPredictor: GRU {num_gru_layers} layers, "
              f"hidden_dim={hidden_dim}, condition_dim={condition_dim}")

    def encode_history(self, pressure_features: torch.Tensor):
        gru_out, h = self.gru(pressure_features)  # h: [num_layers, B, D]
        last_feat = gru_out[:, -1, :]  # [B, D]
        return h, last_feat

    def step(
        self,
        h: torch.Tensor,
        last_feat: torch.Tensor,
        state: torch.Tensor,
        flow_context: torch.Tensor,
        re_embed: torch.Tensor,
    ):

        cond = self.condition_proj(
            torch.cat([state, flow_context, re_embed], dim=-1)
        )  # [B, D]

        gru_input = (last_feat + cond).unsqueeze(1)  # [B, 1, D]
        gru_out, h_new = self.gru(gru_input, h)
        gru_out = gru_out.squeeze(1)  # [B, D]

        combined = torch.cat([gru_out, cond], dim=-1)  # [B, 2D]
        p_feat = self.output_net(combined)  # [B, D]

        return p_feat, h_new, gru_out

class EnhancedPressureDynamicsPredictor(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 512,
        re_dim: int = 64,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
        num_attn_heads: int = 4,
        memory_len: int = 50,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.memory_len = memory_len

        condition_dim = 4 + hidden_dim + re_dim
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )

        self.history_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        print(f"  EnhancedPressureDynamicsPredictor: GRU {num_gru_layers}L, "
              f"Attn {num_attn_heads}H, memory={memory_len}, "
              f"hidden={hidden_dim}, condition={condition_dim}")

    def encode_history(self, pressure_features: torch.Tensor):
        gru_out, h = self.gru(pressure_features)
        last_feat = gru_out[:, -1, :]
        memory = pressure_features
        return h, last_feat, memory

    def step(
        self,
        h: torch.Tensor,
        last_feat: torch.Tensor,
        state: torch.Tensor,
        flow_context: torch.Tensor,
        re_embed: torch.Tensor,
        memory: torch.Tensor,
    ):

        cond = self.condition_proj(
            torch.cat([state, flow_context, re_embed], dim=-1)
        )  # [B, D]

        gate = self.gate_net(
            torch.cat([last_feat, cond], dim=-1)
        )
        gru_input = (gate * last_feat + (1 - gate) * cond).unsqueeze(1)  # [B, 1, D]

        gru_out, h_new = self.gru(gru_input, h)
        gru_out_feat = gru_out.squeeze(1)  # [B, D]

        query = gru_out  # [B, 1, D]
        attn_out, _ = self.history_attn(query, memory, memory)  # [B, 1, D]
        attn_out = self.attn_norm(attn_out.squeeze(1) + gru_out_feat)  # residual + norm

        combined = torch.cat([gru_out_feat, attn_out, cond], dim=-1)  # [B, 3D]
        delta = self.output_net(combined)  # [B, D]

        p_feat = last_feat + self.residual_scale * delta  # [B, D]

        memory_new = torch.cat([memory, gru_out_feat.unsqueeze(1)], dim=1)
        if memory_new.shape[1] > self.memory_len:
            memory_new = memory_new[:, -self.memory_len:, :]

        return p_feat, h_new, gru_out_feat, memory_new

class CausalVIVPredictor(nn.Module):

    def __init__(
        self,

        num_pressure_sensors: int = 72,
        pressure_hidden_dim: int = 512,
        pressure_temporal_layers: int = 6,

        flow_hidden_dim: int = 512,

        re_dim: int = 64,

        num_heads: int = 8,
        dropout: float = 0.1,

        mass: float | None = None,
        stiffness: float | None = None,
        damping: float | None = None,
        dt_data: float | None = None,
        dt_solver: float | None = None,

        num_harmonics: int = 10,
        num_gru_layers: int = 2,
        pressure_encoder_type: str = 'x12',
        sensor_token_dim: int = 96,
        sensor_spatial_layers: int = 2,
        sensor_dropout_prob: float = 0.0,
        context_encoder_type: str = 'mean',
        y_residual_branch: bool = False,
        ring_prior_enabled: bool = False,
        ring_neighbor_alpha: float = 0.10,
        ring_opposite_beta: float = 0.06,

        pressure_dynamics_type: str = 'gru',
        num_attn_heads_pressure: int = 4,
        memory_len: int = 50,
        # Step-1 minimal architecture upgrade: decouple x/y prediction heads
        decouple_xy_heads: bool = False,
        # Step-B2.2: bounded high-Re ODE parameter micro-tuning (x/vx only)
        re_ode_correction_enabled: bool = False,
        re_ode_correction_scale: float = 0.0,
        re_ode_correction_re_threshold: float = 3e5,
        # ODE damping experiment switches
        no_ode_scaffold: bool = False,
        use_blackbox_state: bool = False,
        ode_damping_learnable: bool = False,

        flow_in_channels: int = 3,
        flow_spatial_layers: int = 4,
        flow_temporal_layers: int = 4,
    ):
        super().__init__()

        self.pressure_hidden_dim = pressure_hidden_dim
        self.pressure_dynamics_type = pressure_dynamics_type
        self.pressure_encoder_type = pressure_encoder_type
        self.pressure_encoder_sparse = None
        self.decouple_xy_heads = decouple_xy_heads
        self.y_residual_branch = y_residual_branch
        self.re_ode_correction_enabled = re_ode_correction_enabled
        self.re_ode_correction_scale = float(re_ode_correction_scale)
        self.re_ode_correction_re_threshold = float(re_ode_correction_re_threshold)
        self.no_ode_scaffold = no_ode_scaffold
        self.use_blackbox_state = use_blackbox_state
        self.ode_damping_learnable = ode_damping_learnable

        if pressure_encoder_type == 'x13_sparse':
            self.pressure_encoder = PressureTemporalEncoder_X13(
                num_sensors=num_pressure_sensors,
                hidden_dim=pressure_hidden_dim,
                num_temporal_layers=pressure_temporal_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_harmonics=num_harmonics,
                sensor_token_dim=sensor_token_dim,
                num_spatial_layers=sensor_spatial_layers,
                sensor_dropout_prob=sensor_dropout_prob,
                ring_prior_enabled=ring_prior_enabled,
                ring_neighbor_alpha=ring_neighbor_alpha,
                ring_opposite_beta=ring_opposite_beta,
            )
        elif pressure_encoder_type == 'x13_hybrid':
            self.pressure_encoder = PressureTemporalEncoder_C(
                num_sensors=num_pressure_sensors,
                hidden_dim=pressure_hidden_dim,
                num_temporal_layers=pressure_temporal_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_harmonics=num_harmonics,
            )
            self.pressure_encoder_sparse = PressureTemporalEncoder_X13(
                num_sensors=num_pressure_sensors,
                hidden_dim=pressure_hidden_dim,
                num_temporal_layers=pressure_temporal_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_harmonics=num_harmonics,
                sensor_token_dim=sensor_token_dim,
                num_spatial_layers=sensor_spatial_layers,
                sensor_dropout_prob=sensor_dropout_prob,
                ring_prior_enabled=ring_prior_enabled,
                ring_neighbor_alpha=ring_neighbor_alpha,
                ring_opposite_beta=ring_opposite_beta,
            )
            print("  X13-Hybrid: dense(72 sensors) -> Encoder_C, sparse/masked -> Encoder_X13")
        else:
            self.pressure_encoder = PressureTemporalEncoder_C(
                num_sensors=num_pressure_sensors,
                hidden_dim=pressure_hidden_dim,
                num_temporal_layers=pressure_temporal_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_harmonics=num_harmonics,
            )

        if context_encoder_type == 'attn':
            self.flow_encoder = AttentivePressureContextEncoder(
                hidden_dim=flow_hidden_dim,
                dropout=dropout,
            )
        else:
            self.flow_encoder = PressureContextEncoder(
                hidden_dim=flow_hidden_dim,
                dropout=dropout,
            )

        self.re_embedder = nn.Sequential(
            nn.Linear(1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, re_dim),
            nn.LayerNorm(re_dim),
        )

        self.force_head = ForceHead(
            hidden_dim=pressure_hidden_dim,
            re_dim=re_dim,
            dropout=dropout,
        )

        self.state_correction = StateCorrectionHead(
            hidden_dim=pressure_hidden_dim,
            dropout=dropout,
        )

        if self.decouple_xy_heads:
            refine_in_dim = pressure_hidden_dim + 4 + flow_hidden_dim + re_dim
            self.force_refine_x = nn.Sequential(
                nn.Linear(refine_in_dim, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 1),
            )
            self.force_refine_y = nn.Sequential(
                nn.Linear(refine_in_dim, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 1),
            )

            state_refine_in = pressure_hidden_dim + 4 + 2
            self.state_refine_x = nn.Sequential(
                nn.Linear(state_refine_in, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 2),  # dx, dvx
            )
            self.state_refine_y = nn.Sequential(
                nn.Linear(state_refine_in, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 2),  # dy, dvy
            )

            for m in [self.force_refine_x, self.force_refine_y, self.state_refine_x, self.state_refine_y]:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

        if self.y_residual_branch:
            y_in_dim = pressure_hidden_dim + 4 + flow_hidden_dim + re_dim
            self.y_force_residual = nn.Sequential(
                nn.Linear(y_in_dim, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 1),
            )
            self.y_state_residual = nn.Sequential(
                nn.Linear(y_in_dim, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.y_force_residual[-1].weight)
            nn.init.zeros_(self.y_force_residual[-1].bias)
            nn.init.zeros_(self.y_state_residual[-1].weight)
            nn.init.zeros_(self.y_state_residual[-1].bias)

        if self.re_ode_correction_enabled:
            self.re_ode_corr = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 2),  # delta_k, delta_c
            )
            nn.init.zeros_(self.re_ode_corr[-1].weight)
            nn.init.zeros_(self.re_ode_corr[-1].bias)
        else:
            self.re_ode_corr = None

        self.ode_solver = PhysicsODESolver(
            mass=mass, stiffness=stiffness, damping=damping,
            dt_data=dt_data, dt_solver=dt_solver,
        )

        if self.ode_damping_learnable:
            self.learnable_damping = nn.Parameter(torch.tensor(0.0))
        else:
            self.learnable_damping = None

        if self.use_blackbox_state:
            self.blackbox_state_updater = nn.Sequential(
                nn.Linear(4 + 2 + pressure_hidden_dim, pressure_hidden_dim),
                nn.LayerNorm(pressure_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pressure_hidden_dim, pressure_hidden_dim // 2),
                nn.LayerNorm(pressure_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(pressure_hidden_dim // 2, 4),
            )
        else:
            self.blackbox_state_updater = None

        if pressure_dynamics_type == 'enhanced':
            self.pressure_dynamics = EnhancedPressureDynamicsPredictor(
                hidden_dim=pressure_hidden_dim,
                re_dim=re_dim,
                num_gru_layers=num_gru_layers,
                dropout=dropout,
                num_attn_heads=num_attn_heads_pressure,
                memory_len=memory_len,
            )
        else:
            self.pressure_dynamics = PressureDynamicsPredictor(
                hidden_dim=pressure_hidden_dim,
                re_dim=re_dim,
                num_gru_layers=num_gru_layers,
                dropout=dropout,
            )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"CausalVIVPredictor : {total_params / 1e6:.2f}M")
        print(f"  decouple_xy_heads={self.decouple_xy_heads}")
        print(f"  y_residual_branch={self.y_residual_branch}, ring_prior_enabled={ring_prior_enabled}")
        print(
            f"  re_ode_correction_enabled={self.re_ode_correction_enabled}, "
            f"re_ode_correction_scale={self.re_ode_correction_scale}, "
            f"re_ode_correction_re_threshold={self.re_ode_correction_re_threshold}"
        )

    def _is_full_sensor_mask(self, sensor_mask: Optional[torch.Tensor]) -> bool:
        if sensor_mask is None:
            return True
        if sensor_mask.numel() == 0:
            return True
        return bool((sensor_mask >= 0.999).all().item())

    def _should_use_sparse_encoder(self, sensor_mask: Optional[torch.Tensor]) -> bool:
        if self.pressure_encoder_type == 'x13_sparse':
            return True
        if self.pressure_encoder_type != 'x13_hybrid':
            return False
        if self.pressure_encoder_sparse is None:
            return False
        sparse_dropout = getattr(self.pressure_encoder_sparse, 'sensor_dropout_prob', 0.0)
        if self.training and sparse_dropout > 0:
            return True
        return not self._is_full_sensor_mask(sensor_mask)

    def _get_sparse_encoder(self):
        if self.pressure_encoder_type == 'x13_sparse':
            return self.pressure_encoder
        return self.pressure_encoder_sparse

    def _encoder_accepts_sensor_mask(self, encoder) -> bool:
        if encoder is None:
            return False
        try:
            sig = inspect.signature(encoder.forward)
            return 'sensor_mask' in sig.parameters
        except Exception:
            return False

    def _encode_pressure_sequence(
        self,
        pressure_seq: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._should_use_sparse_encoder(sensor_mask):
            sparse_encoder = self._get_sparse_encoder()
            if sparse_encoder is not None:
                if self._encoder_accepts_sensor_mask(sparse_encoder):
                    return sparse_encoder(pressure_seq, sensor_mask=sensor_mask)
                if self._encoder_accepts_sensor_mask(self.pressure_encoder_sparse):
                    return self.pressure_encoder_sparse(pressure_seq, sensor_mask=sensor_mask)
        return self.pressure_encoder(pressure_seq)

    def _encode_pressure_frame(
        self,
        pressure: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._should_use_sparse_encoder(sensor_mask):
            sparse_encoder = self._get_sparse_encoder()
            if sparse_encoder is not None:
                if self._encoder_accepts_sensor_mask(sparse_encoder):
                    return sparse_encoder.encode_frame_features(pressure, sensor_mask=sensor_mask)
                if self._encoder_accepts_sensor_mask(self.pressure_encoder_sparse):
                    return self.pressure_encoder_sparse.encode_frame_features(pressure, sensor_mask=sensor_mask)

        enc = self.pressure_encoder
        if hasattr(enc, 'encode_frame_features'):
            return enc.encode_frame_features(pressure, sensor_mask=sensor_mask)
        main_feat = enc.frame_encoder(pressure)
        fourier_coeffs = torch.matmul(pressure, enc.fourier_basis.T)
        physics_feat = enc.physics_net(fourier_coeffs)
        return main_feat + enc.physics_scale * physics_feat

    def forward(
        self,
        surface_pressure_input: torch.Tensor,
        reynolds_number: torch.Tensor,            # [B]
        initial_state: torch.Tensor,              # [B, 4]
        T_out: int = 10,

        true_states: Optional[torch.Tensor] = None,    # [B, T_out, 4]
        teacher_forcing_ratio: float = 0.0,

        true_pressure_output: Optional[torch.Tensor] = None,  # [B, T_out, 72]
        pressure_tf_ratio: float = 0.0,
        sensor_mask: Optional[torch.Tensor] = None,           # [B,72] or [B,T,72]

        flow_field_input: Optional[torch.Tensor] = None,
        distill_alignment: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = initial_state.shape[0]

        if distill_alignment and self.pressure_encoder_type == 'x13_hybrid':
            with torch.no_grad():
                dense_p_features = self.pressure_encoder(surface_pressure_input)
            p_features = self.pressure_encoder_sparse(surface_pressure_input, sensor_mask=sensor_mask)
            distill_loss = torch.nn.functional.mse_loss(p_features, dense_p_features)
        else:
            p_features = self._encode_pressure_sequence(surface_pressure_input, sensor_mask=sensor_mask)
            distill_loss = torch.tensor(0.0, device=p_features.device)

        flow_context = self.flow_encoder(p_features)  # [B, D]

        log_re = torch.log10(reynolds_number.unsqueeze(-1))
        re_embed = self.re_embedder(log_re)  # [B, re_dim]
        high_re_gate = (reynolds_number.unsqueeze(-1) >= self.re_ode_correction_re_threshold).float()

        if self.pressure_dynamics_type == 'enhanced':
            gru_h, last_p_feat, memory = self.pressure_dynamics.encode_history(p_features)
        else:
            gru_h, last_p_feat = self.pressure_dynamics.encode_history(p_features)
            memory = None

        p_tf_targets = None
        if self.training and true_pressure_output is not None and pressure_tf_ratio > 0:
            with torch.no_grad():
                p_tf_targets = self._encode_pressure_frame(
                    true_pressure_output,
                    sensor_mask=sensor_mask,
                )  # [B, T_out, D]

        all_forces = []
        all_states = []
        all_p_feats = []
        current_state = initial_state

        x_eq = initial_state[:, 0]
        y_eq = initial_state[:, 1]

        use_state_tf = self.training and true_states is not None and teacher_forcing_ratio > 0

        for t in range(T_out):

            if self.pressure_dynamics_type == 'enhanced':
                p_feat_t, gru_h, last_p_feat, memory = self.pressure_dynamics.step(
                    gru_h, last_p_feat, current_state, flow_context, re_embed, memory
                )
            else:
                p_feat_t, gru_h, last_p_feat = self.pressure_dynamics.step(
                    gru_h, last_p_feat, current_state, flow_context, re_embed
                )

            if self.training and p_tf_targets is not None and pressure_tf_ratio > 0:
                B = p_feat_t.shape[0]
                p_tf_mask = (torch.rand(B, 1, device=p_feat_t.device) < pressure_tf_ratio)
                p_feat_t = torch.where(p_tf_mask.expand_as(p_feat_t),
                                       p_tf_targets[:, t, :], p_feat_t)

            all_p_feats.append(p_feat_t)

            force_t = self.force_head(
                pressure_feat=p_feat_t,
                state=current_state,
                flow_context=flow_context,
                re_embed=re_embed,
            )  # [B, 2]

            if self.decouple_xy_heads:
                refine_feat = torch.cat([p_feat_t, current_state, flow_context, re_embed], dim=-1)
                dfx = self.force_refine_x(refine_feat)
                dfy = self.force_refine_y(refine_feat)
                force_t = force_t + torch.cat([dfx, dfy], dim=-1)

            if self.y_residual_branch:
                y_feat = torch.cat([p_feat_t, current_state, flow_context, re_embed], dim=-1)
                force_t = force_t.clone()
                force_t[:, 1:2] = force_t[:, 1:2] + self.y_force_residual(y_feat)

            if self.re_ode_correction_enabled and self.re_ode_corr is not None and self.re_ode_correction_scale > 0:
                re_dyn_in = torch.cat([
                    log_re,
                    current_state[:, 2:3],  # vx
                    force_t[:, 0:1],        # Fx
                ], dim=-1)
                dk_dc = torch.tanh(self.re_ode_corr(re_dyn_in)) * self.re_ode_correction_scale  # [B,2]
                dk_dc = dk_dc * high_re_gate

                x_centered = current_state[:, 0:1] - x_eq.unsqueeze(-1)
                fx_corr = dk_dc[:, 0:1] * x_centered + dk_dc[:, 1:2] * current_state[:, 2:3]
                force_t = force_t.clone()
                force_t[:, 0:1] = force_t[:, 0:1] - fx_corr

            if self.no_ode_scaffold:

                state_ode = current_state
            elif self.use_blackbox_state:

                bb_input = torch.cat([current_state, force_t, p_feat_t], dim=-1)
                state_ode = current_state + self.blackbox_state_updater(bb_input)
            elif self.ode_damping_learnable and self.learnable_damping is not None:

                c_eff = torch.nn.functional.softplus(self.learnable_damping)
                state_ode = self.ode_solver.step(current_state, force_t, x_eq=x_eq, y_eq=y_eq, c_override=c_eff)
            else:

                state_ode = self.ode_solver.step(current_state, force_t, x_eq=x_eq, y_eq=y_eq)
            next_state = self.state_correction(state_ode, force_t, p_feat_t)

            if self.decouple_xy_heads:
                s_refine_feat = torch.cat([state_ode, force_t, p_feat_t], dim=-1)
                x_pair = self.state_refine_x(s_refine_feat)  # dx, dvx
                y_pair = self.state_refine_y(s_refine_feat)  # dy, dvy
                state_delta = torch.stack([x_pair[:, 0], y_pair[:, 0], x_pair[:, 1], y_pair[:, 1]], dim=-1)
                next_state = next_state + state_delta

            if self.y_residual_branch:
                y_feat = torch.cat([p_feat_t, current_state, flow_context, re_embed], dim=-1)
                next_state = next_state.clone()
                next_state[:, 1:2] = next_state[:, 1:2] + self.y_state_residual(y_feat)

            all_forces.append(force_t)
            all_states.append(next_state)

            if use_state_tf and t < T_out - 1:
                B = next_state.shape[0]
                s_tf_mask = (torch.rand(B, 1, device=next_state.device) < teacher_forcing_ratio)
                current_state = torch.where(s_tf_mask.expand_as(next_state),
                                            true_states[:, t, :], next_state)
            else:
                current_state = next_state

        return {
            'forces': torch.stack(all_forces, dim=1),        # [B, T_out, 2]
            'states': torch.stack(all_states, dim=1),        # [B, T_out, 4]
            'pred_pressure_features': torch.stack(all_p_feats, dim=1),  # [B, T_out, D]
            'distill_loss': distill_loss,
        }

    def load_oracle_weights(self, oracle_checkpoint_path: str, device='cpu'):
        ckpt = torch.load(oracle_checkpoint_path, map_location=device, weights_only=False)
        oracle_state = ckpt['model_state_dict']

        transferable_prefixes = [
            'pressure_encoder.', 'flow_encoder.',  # flow_encoder = PressureContextEncoder (V6)
            're_embedder.', 'force_head.', 'state_correction.', 'ode_solver.',
        ]

        my_state = self.state_dict()
        transferred = 0
        skipped = 0

        for key, value in oracle_state.items():
            if any(key.startswith(p) for p in transferable_prefixes):
                if key in my_state and my_state[key].shape == value.shape:
                    my_state[key] = value
                    transferred += 1
                else:
                    skipped += 1

        self.load_state_dict(my_state)
        print(f"  Oracle: {transferred} transferred, {skipped} skipped, "
              f"pressure_dynamics")
        return transferred

# Loss Function
def compute_causal_viv_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    weights: Dict[str, float] = None,
    model: CausalVIVPredictor = None,
) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = {
            'force': 1.0,
            'position': 5.0,
            'velocity': 2.0,
            'variance': 0.2,
            'smoothness': 0.02,
            'pressure_feat': 0.1,
            'cumulative_pos': 0.8,
            'spectral': 0.3,
            'dc_x': 2.0,
            'ac_x': 1.0,
        }

    forces_pred = pred['forces']
    states_pred = pred['states']
    forces_true = batch['force_output'].to(device)
    states_true = batch['state_output'].to(device)

    T_out = forces_pred.shape[1]

    temporal_alpha = weights.get('temporal_alpha', 1.0)
    t_indices = torch.arange(T_out, device=device, dtype=torch.float32)
    t_weights = 1.0 + temporal_alpha * t_indices / T_out
    t_weights = t_weights / t_weights.mean()

    force_x_err = torch.abs(forces_pred[:, :, 0] - forces_true[:, :, 0])
    force_y_err = torch.abs(forces_pred[:, :, 1] - forces_true[:, :, 1])
    force_x_loss = (force_x_err * t_weights.unsqueeze(0)).mean()
    force_y_loss = (force_y_err * t_weights.unsqueeze(0)).mean()
    force_loss = 0.5 * force_x_loss + 0.5 * force_y_loss

    pos_pred = states_pred[:, :, :2]
    pos_true = states_true[:, :, :2]
    pos_x_err = torch.abs(pos_pred[:, :, 0] - pos_true[:, :, 0])
    pos_y_err = torch.abs(pos_pred[:, :, 1] - pos_true[:, :, 1])
    pos_x_loss = (pos_x_err * t_weights.unsqueeze(0)).mean()
    pos_y_loss = (pos_y_err * t_weights.unsqueeze(0)).mean()
    pos_loss = 0.5 * pos_x_loss + 0.5 * pos_y_loss

    # ---- 3. Velocity Loss ----
    vel_pred = states_pred[:, :, 2:]
    vel_true = states_true[:, :, 2:]
    vel_err = torch.abs(vel_pred - vel_true)
    vel_x_loss = (vel_err[:, :, 0] * t_weights.unsqueeze(0)).mean()
    vel_y_loss = (vel_err[:, :, 1] * t_weights.unsqueeze(0)).mean()
    vel_loss = 0.5 * vel_x_loss + 0.5 * vel_y_loss

    dc_x_loss = torch.tensor(0.0, device=device)
    if weights.get('dc_x', 0) > 0:

        fx_mean_pred = forces_pred[:, :, 0].mean(dim=1)   # [B]
        fx_mean_true = forces_true[:, :, 0].mean(dim=1)   # [B]
        dc_fx_loss = torch.abs(fx_mean_pred - fx_mean_true).mean()

        px_mean_pred = pos_pred[:, :, 0].mean(dim=1)      # [B]
        px_mean_true = pos_true[:, :, 0].mean(dim=1)      # [B]
        dc_px_loss = torch.abs(px_mean_pred - px_mean_true).mean()

        dc_x_loss = dc_fx_loss + 5.0 * dc_px_loss

    ac_x_loss = torch.tensor(0.0, device=device)
    if weights.get('ac_x', 0) > 0:

        fx_ac_pred = forces_pred[:, :, 0] - forces_pred[:, :, 0].mean(dim=1, keepdim=True)
        fx_ac_true = forces_true[:, :, 0] - forces_true[:, :, 0].mean(dim=1, keepdim=True).detach()
        fx_ac_loss = torch.abs(fx_ac_pred - fx_ac_true).mean()

        px_ac_pred = pos_pred[:, :, 0] - pos_pred[:, :, 0].mean(dim=1, keepdim=True)
        px_ac_true = pos_true[:, :, 0] - pos_true[:, :, 0].mean(dim=1, keepdim=True).detach()
        px_ac_loss = torch.abs(px_ac_pred - px_ac_true).mean()
        ac_x_loss = fx_ac_loss + 5.0 * px_ac_loss

    relative_traj_x_loss = torch.tensor(0.0, device=device)
    if weights.get('relative_traj_x', 0) > 0:
        init_state = batch.get('initial_state', None)
        if init_state is not None:
            init_x = init_state[:, 0:1].to(device)
            init_vx = init_state[:, 2:3].to(device)

            dx_pred = pos_pred[:, :, 0] - init_x        # [B, T]
            dx_true = pos_true[:, :, 0] - init_x        # [B, T]

            fx_dc = forces_true[:, :, 0].mean(dim=1, keepdim=True).detach()
            fx_rel_pred = forces_pred[:, :, 0] - fx_dc
            fx_rel_true = forces_true[:, :, 0] - fx_dc
            relative_traj_x_loss = (
                torch.abs(dx_pred - dx_true).mean()
                + 0.5 * torch.abs(fx_rel_pred - fx_rel_true).mean()
            )

    vx_emphasis_loss = torch.tensor(0.0, device=device)
    if weights.get('vx_emphasis', 0) > 0:
        vx_err = torch.abs(vel_pred[:, :, 0] - vel_true[:, :, 0])
        vx_emphasis_loss = (vx_err * t_weights.unsqueeze(0)).mean()

    eps = 1e-6
    pred_std_fx = torch.std(forces_pred[:, :, 0]).clamp(min=eps)
    pred_std_fy = torch.std(forces_pred[:, :, 1]).clamp(min=eps)
    true_std_fx = torch.std(forces_true[:, :, 0]).detach().clamp(min=eps)
    true_std_fy = torch.std(forces_true[:, :, 1]).detach().clamp(min=eps)
    var_loss_fx = torch.abs(torch.log(pred_std_fx) - torch.log(true_std_fx))
    var_loss_fy = torch.abs(torch.log(pred_std_fy) - torch.log(true_std_fy))
    variance_loss = 0.5 * var_loss_fx + 0.5 * var_loss_fy

    # ---- 7. Smoothness Regularization ----
    pos_diff2 = torch.diff(torch.diff(pos_pred, dim=1), dim=1)
    smoothness_loss = torch.mean(pos_diff2 ** 2)

    x_diff2 = torch.diff(torch.diff(pos_pred[:, :, 0], dim=1), dim=1)  # [B, T-2]
    x_smooth_loss = x_diff2.abs().mean()

    #   x: ~ 5e-4,  y: ~ 5e-3,  vx: ~ 5e-5,  vy: ~ 5e-4
    norm_pos_loss = torch.tensor(0.0, device=device)
    norm_vel_loss = torch.tensor(0.0, device=device)
    if weights.get('norm_pos', 0) > 0:
        ref_px, ref_py = 5e-4, 5e-3
        norm_px = torch.log1p((pos_pred[:, :, 0] - pos_true[:, :, 0]).pow(2) / ref_px ** 2).mean()
        norm_py = torch.log1p((pos_pred[:, :, 1] - pos_true[:, :, 1]).pow(2) / ref_py ** 2).mean()
        norm_pos_loss = 0.5 * norm_px + 0.5 * norm_py
    if weights.get('norm_vel', 0) > 0:
        ref_vx, ref_vy = 5e-5, 5e-4
        norm_vx = torch.log1p((vel_pred[:, :, 0] - vel_true[:, :, 0]).pow(2) / ref_vx ** 2).mean()
        norm_vy = torch.log1p((vel_pred[:, :, 1] - vel_true[:, :, 1]).pow(2) / ref_vy ** 2).mean()
        norm_vel_loss = 0.5 * norm_vx + 0.5 * norm_vy

    norm_force_x_loss = torch.tensor(0.0, device=device)
    if weights.get('norm_force_x', 0) > 0:

        fx_std_ps = forces_true[:, :, 0].std(dim=1, keepdim=True).clamp(min=2e-4)   # [B,1]
        nfx_raw = torch.abs(forces_pred[:, :, 0] - forces_true[:, :, 0]) / fx_std_ps
        nfx_loss = nfx_raw.clamp(max=10.0).mean()

        px_std_ps = pos_true[:, :, 0].std(dim=1, keepdim=True).clamp(min=1e-6)      # [B,1]
        npx_raw = torch.abs(pos_pred[:, :, 0] - pos_true[:, :, 0]) / px_std_ps
        npx_loss = npx_raw.clamp(max=10.0).mean()
        norm_force_x_loss = nfx_loss + npx_loss
        norm_force_x_loss = torch.nan_to_num(norm_force_x_loss, nan=0.0)

    norm_pos_adaptive_loss = torch.tensor(0.0, device=device)
    if weights.get('norm_pos_adaptive', 0) > 0:

        px_std_ps = pos_true[:, :, 0].std(dim=1, keepdim=True).clamp(min=1e-6)
        py_std_ps = pos_true[:, :, 1].std(dim=1, keepdim=True).clamp(min=1e-6)
        adp_px = torch.log1p((pos_pred[:, :, 0] - pos_true[:, :, 0]).pow(2)
                             / px_std_ps.pow(2).clamp(min=1e-12)).clamp(max=20.0).mean()
        adp_py = torch.log1p((pos_pred[:, :, 1] - pos_true[:, :, 1]).pow(2)
                             / py_std_ps.pow(2).clamp(min=1e-12)).clamp(max=20.0).mean()
        norm_pos_adaptive_loss = 0.5 * adp_px + 0.5 * adp_py
        norm_pos_adaptive_loss = torch.nan_to_num(norm_pos_adaptive_loss, nan=0.0)

    x_pearson_loss = torch.tensor(0.0, device=device)
    if weights.get('x_pearson', 0) > 0:
        x_pred_c = pos_pred[:, :, 0] - pos_pred[:, :, 0].mean(dim=1, keepdim=True)
        x_true_c = pos_true[:, :, 0] - pos_true[:, :, 0].mean(dim=1, keepdim=True)
        x_pred_n = x_pred_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_true_n = x_true_c.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_corr = (x_pred_c / x_pred_n * x_true_c / x_true_n).sum(dim=1)  # [B]
        x_pearson_loss = (1.0 - x_corr).mean()

    x_peak_loss = torch.tensor(0.0, device=device)
    if weights.get('x_peak', 0) > 0:
        fx_amp_pred = forces_pred[:, :, 0].amax(dim=1) - forces_pred[:, :, 0].amin(dim=1)
        fx_amp_true = forces_true[:, :, 0].amax(dim=1) - forces_true[:, :, 0].amin(dim=1)
        px_amp_pred = pos_pred[:, :, 0].amax(dim=1) - pos_pred[:, :, 0].amin(dim=1)
        px_amp_true = pos_true[:, :, 0].amax(dim=1) - pos_true[:, :, 0].amin(dim=1)
        x_peak_loss = torch.abs(fx_amp_pred - fx_amp_true).mean() + 5.0 * torch.abs(px_amp_pred - px_amp_true).mean()

    # ---- 8. Pressure Feature Auxiliary Loss ----
    pressure_feat_loss = torch.tensor(0.0, device=device)
    if (model is not None
        and weights.get('pressure_feat', 0) > 0
        and 'pred_pressure_features' in pred
        and 'surface_pressure_output' in batch):
        true_pressure = batch['surface_pressure_output'].to(device)
        with torch.no_grad():
            target_feats = model._encode_pressure_frame(true_pressure)
        pred_feats = pred['pred_pressure_features']
        cos_sim = F.cosine_similarity(
            pred_feats.reshape(-1, pred_feats.shape[-1]),
            target_feats.reshape(-1, target_feats.shape[-1]),
            dim=-1,
        )
        pressure_feat_loss = 1.0 - cos_sim.mean()

    cumulative_pos_loss = torch.tensor(0.0, device=device)
    if weights.get('cumulative_pos', 0) > 0:
        pos_error = pos_pred - pos_true
        cum_error = torch.cumsum(pos_error, dim=1)
        cum_x = torch.abs(cum_error[:, :, 0]).mean()
        cum_y = torch.abs(cum_error[:, :, 1]).mean()
        cumulative_pos_loss = 0.5 * cum_x + 0.5 * cum_y

    spectral_loss = torch.tensor(0.0, device=device)
    if weights.get('spectral', 0) > 0 and T_out >= 4:
        def spectral_mag_loss(pred_sig, true_sig):
            pred_fft = torch.fft.rfft(pred_sig.float(), dim=1)
            true_fft = torch.fft.rfft(true_sig.float(), dim=1)
            return torch.abs(torch.abs(pred_fft) - torch.abs(true_fft)).mean()

        fy_spec = spectral_mag_loss(forces_pred[:, :, 1], forces_true[:, :, 1])
        py_spec = spectral_mag_loss(pos_pred[:, :, 1], pos_true[:, :, 1])

        fx_spec = spectral_mag_loss(forces_pred[:, :, 0], forces_true[:, :, 0])
        px_spec = spectral_mag_loss(pos_pred[:, :, 0], pos_true[:, :, 0])

        spectral_loss = 0.35 * fy_spec + 0.35 * fx_spec + 0.15 * py_spec + 0.15 * px_spec
        spectral_loss = torch.nan_to_num(spectral_loss, nan=0.0)

    total = (
        weights['force']                          * force_loss +
        weights['position']                       * pos_loss +
        weights['velocity']                       * vel_loss +
        weights['variance']                       * variance_loss +
        weights['smoothness']                     * smoothness_loss +
        weights.get('pressure_feat', 0)           * pressure_feat_loss +
        weights.get('cumulative_pos', 0)          * cumulative_pos_loss +
        weights.get('spectral', 0)                * spectral_loss +
        weights.get('dc_x', 0)                    * dc_x_loss +
        weights.get('ac_x', 0)                    * ac_x_loss +
        weights.get('relative_traj_x', 0)         * relative_traj_x_loss +
        weights.get('vx_emphasis', 0)             * vx_emphasis_loss +
        weights.get('x_smooth', 0)                * x_smooth_loss +
        weights.get('norm_pos', 0)                * norm_pos_loss +
        weights.get('norm_vel', 0)                * norm_vel_loss +
        weights.get('x_pearson', 0)               * x_pearson_loss +
        weights.get('x_peak', 0)                  * x_peak_loss +
        weights.get('norm_force_x', 0)            * norm_force_x_loss +
        weights.get('norm_pos_adaptive', 0)       * norm_pos_adaptive_loss +
        weights.get('distill_align', 0)           * pred.get('distill_loss', torch.tensor(0.0, device=device))
    )
    total = torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=-1e4)

    return {
        'total': total,
        'force': force_loss,
        'force_x': force_x_loss,
        'force_y': force_y_loss,
        'position': pos_loss,
        'velocity': vel_loss,
        'variance': variance_loss,
        'smoothness': smoothness_loss,
        'pressure_feat': pressure_feat_loss,
        'cumulative_pos': cumulative_pos_loss,
        'distill_align': pred.get('distill_loss', torch.tensor(0.0, device=device)),
        'spectral': spectral_loss,
        'dc_x': dc_x_loss,
        'ac_x': ac_x_loss,
        'rel_traj_x': relative_traj_x_loss,
        'vx_emph': vx_emphasis_loss,
        'x_smooth': x_smooth_loss,
        'norm_pos': norm_pos_loss,
        'norm_vel': norm_vel_loss,
        'x_pearson': x_pearson_loss,
        'x_peak': x_peak_loss,
        'norm_force_x': norm_force_x_loss,
        'norm_pos_adp': norm_pos_adaptive_loss,
    }
