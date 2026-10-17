import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# ODE Solver (Physics-based state integration)
class PhysicsODESolver(nn.Module):
    def __init__(self, mass, stiffness, damping, dt_data, dt_solver):
        super().__init__()
        if any(x is None for x in (mass, stiffness, damping, dt_data, dt_solver)):
            raise ValueError('Supply actual-model values for mass, stiffness, damping, dt_data and dt_solver')
        if mass <= 0 or stiffness <= 0 or damping < 0 or dt_data <= 0 or dt_solver <= 0:
            raise ValueError('ODE parameters must be positive, except damping may be zero')
        self.register_buffer('mass', torch.tensor(mass, dtype=torch.float32))
        self.register_buffer('stiffness', torch.tensor(stiffness, dtype=torch.float32))
        self.register_buffer('damping', torch.tensor(damping, dtype=torch.float32))
        self.dt_data = dt_data
        self.dt_solver = dt_solver
        self.substeps = max(1, int(round(dt_data / dt_solver)))

        print(f"PhysicsODESolver: m={mass:.3f}, k={stiffness:.3f}, c={damping:.3f}, dt_data={dt_data}, substeps={self.substeps}")

    def dynamics(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
        x_eq: torch.Tensor | None = None,
        y_eq: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, y = state[:, 0], state[:, 1]
        vx, vy = state[:, 2], state[:, 3]
        Fx, Fy = force[:, 0], force[:, 1]

        x_ref = x_eq if x_eq is not None else torch.zeros_like(x)
        y_ref = y_eq if y_eq is not None else torch.zeros_like(y)

        c_eff = c_override if c_override is not None else self.damping

        ax = (Fx - self.stiffness * (x - x_ref) - c_eff * vx) / self.mass
        ay = (Fy - self.stiffness * (y - y_ref) - c_eff * vy) / self.mass

        return torch.stack([vx, vy, ax, ay], dim=-1)

    def rk4_step(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
        dt: float,
        x_eq: torch.Tensor | None = None,
        y_eq: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k1 = self.dynamics(state,                 force, x_eq, y_eq, c_override)
        k2 = self.dynamics(state + 0.5 * dt * k1, force, x_eq, y_eq, c_override)
        k3 = self.dynamics(state + 0.5 * dt * k2, force, x_eq, y_eq, c_override)
        k4 = self.dynamics(state + dt * k3,        force, x_eq, y_eq, c_override)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(
        self,
        state: torch.Tensor,
        force: torch.Tensor,
        x_eq: torch.Tensor | None = None,
        y_eq: torch.Tensor | None = None,
        c_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for _ in range(self.substeps):
            state = self.rk4_step(state, force, self.dt_solver, x_eq, y_eq, c_override)
        return state

# Pressure Temporal Encoder
class PressureTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_sensors: int = 72,
        hidden_dim: int = 256,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Step 1: Per-frame encoding (72 sensors → hidden_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(num_sensors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Step 2: Temporal position encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 200, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Step 3: Temporal Transformer (bidirectional OK - we have all frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        B, T, _ = pressure.shape

        # Per-frame encoding
        feat = self.frame_encoder(pressure)  # [B, T, hidden_dim]

        # Add positional encoding
        feat = feat + self.pos_embed[:, :T, :]

        # Temporal attention
        feat = self.temporal_transformer(feat)  # [B, T, hidden_dim]

        return feat

# Flow Context Encoder
class FlowContextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_spatial_layers: int = 3,
        num_temporal_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        from models.spatiotemporal_transformer_encoder import SpatioTemporalTransformerEncoder

        self.encoder = SpatioTemporalTransformerEncoder(
            in_channels=in_channels,
            embed_dim=hidden_dim,
            spatial_depth=num_spatial_layers,
            temporal_depth=num_temporal_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Pool temporal sequence → single context vector
        self.temporal_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, flow_field: torch.Tensor) -> torch.Tensor:
        # SpatioTemporalTransformerEncoder returns [B, T, hidden_dim]
        features = self.encoder(flow_field)  # [B, T_in, hidden_dim]

        # Global temporal average pooling → context vector
        context = features.mean(dim=1)  # [B, hidden_dim]
        context = self.temporal_pool(context)

        return context

class PressureContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        print(f"  PressureContextEncoder: hidden_dim={hidden_dim} (replaces FlowContextEncoder)")

    def forward(self, pressure_features: torch.Tensor) -> torch.Tensor:
        # Temporal average pooling → single context vector
        pooled = pressure_features.mean(dim=1)  # [B, D]
        context = self.context_proj(pooled)
        return context

class AttentivePressureContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        print(f"  AttentivePressureContextEncoder: hidden_dim={hidden_dim}")

    def forward(self, pressure_features: torch.Tensor) -> torch.Tensor:
        logits = self.score_net(pressure_features).squeeze(-1)          # [B, T]
        weights = torch.softmax(logits, dim=1)                          # [B, T]
        pooled = torch.sum(weights.unsqueeze(-1) * pressure_features, dim=1)
        return self.context_proj(pooled)

# Force Prediction Head
class ForceHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        re_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = hidden_dim + 4 + hidden_dim  # pressure + state + flow (Re via FiLM)

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm3 = nn.LayerNorm(hidden_dim // 2)
        self.out_layer = nn.Linear(hidden_dim // 2, 2)  # Fx, Fy

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # layer1, layer2: hidden_dim; layer3: hidden_dim//2
        self.film1 = nn.Linear(re_dim, hidden_dim * 2)   # γ1, β1
        self.film2 = nn.Linear(re_dim, hidden_dim * 2)   # γ2, β2
        self.film3 = nn.Linear(re_dim, (hidden_dim // 2) * 2)  # γ3, β3

        for film in [self.film1, self.film2, self.film3]:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        nn.init.uniform_(self.out_layer.weight, -0.01, 0.01)
        nn.init.zeros_(self.out_layer.bias)

    def _film_modulate(self, h, film_params):
        """FiLM: h' = (1 + γ) ⊙ h + β"""
        gamma, beta = film_params.chunk(2, dim=-1)
        return (1 + gamma) * h + beta

    def forward(
        self,
        pressure_feat: torch.Tensor,
        state: torch.Tensor,
        flow_context: torch.Tensor,
        re_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            force: [B, 2]
        """
        x = torch.cat([pressure_feat, state, flow_context], dim=-1)

        # Layer 1 + FiLM
        h = self.layer1(x)
        h = self._film_modulate(h, self.film1(re_embed))
        h = self.drop(self.act(self.norm1(h)))

        # Layer 2 + FiLM
        h = self.layer2(h)
        h = self._film_modulate(h, self.film2(re_embed))
        h = self.drop(self.act(self.norm2(h)))

        # Layer 3 + FiLM
        h = self.layer3(h)
        h = self._film_modulate(h, self.film3(re_embed))
        h = self.act(self.norm3(h))

        return self.out_layer(h)

class StateCorrectionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = 4 + 2 + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),  # dx, dy, dvx, dvy
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        state_ode: torch.Tensor,
        force: torch.Tensor,
        pressure_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_ode, force, pressure_feat], dim=-1)
        residual = self.net(x)
        return state_ode + residual

# Main Model: VIVPressurePredictor
class VIVPressurePredictor(nn.Module):
    def __init__(
        self,

        num_pressure_sensors: int = 72,
        pressure_hidden_dim: int = 256,
        pressure_temporal_layers: int = 4,

        flow_in_channels: int = 3,
        flow_hidden_dim: int = 256,
        flow_spatial_layers: int = 3,
        flow_temporal_layers: int = 3,

        re_dim: int = 64,

        num_heads: int = 8,
        dropout: float = 0.1,

        mass: float | None = None,
        stiffness: float | None = None,
        damping: float | None = None,
        dt_data: float | None = None,
        dt_solver: float | None = None,
    ):
        super().__init__()

        self.pressure_encoder = PressureTemporalEncoder(
            num_sensors=num_pressure_sensors,
            hidden_dim=pressure_hidden_dim,
            num_temporal_layers=pressure_temporal_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.flow_encoder = FlowContextEncoder(
            in_channels=flow_in_channels,
            hidden_dim=flow_hidden_dim,
            num_spatial_layers=flow_spatial_layers,
            num_temporal_layers=flow_temporal_layers,
            num_heads=num_heads,
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
            hidden_dim=pressure_hidden_dim,  # pressure_hidden_dim == flow_hidden_dim
            re_dim=re_dim,
            dropout=dropout,
        )

        self.state_correction = StateCorrectionHead(
            hidden_dim=pressure_hidden_dim,
            dropout=dropout,
        )

        self.ode_solver = PhysicsODESolver(
            mass=mass,
            stiffness=stiffness,
            damping=damping,
            dt_data=dt_data,
            dt_solver=dt_solver,
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"VIVPressurePredictor : {total_params / 1e6:.2f}M")

    def forward(
        self,
        flow_field_input: torch.Tensor,       # [B, T_in, 3, H, W]
        surface_pressure_full: torch.Tensor,
        reynolds_number: torch.Tensor,        # [B]
        initial_state: torch.Tensor,
        T_out: int = 10,
        true_states: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        B = initial_state.shape[0]
        T_total = surface_pressure_full.shape[1]
        T_in = T_total - T_out

        p_features = self.pressure_encoder(surface_pressure_full)  # [B, T_total, pressure_hidden_dim]

        flow_context = self.flow_encoder(flow_field_input)  # [B, flow_hidden_dim]

        re_embed = self.re_embedder(torch.log10(reynolds_number.unsqueeze(-1)))  # [B, re_dim]

        all_forces = []
        all_states = []
        current_state = initial_state  # [B, 4]

        use_tf = self.training and true_states is not None and teacher_forcing_ratio > 0

        for t in range(T_out):

            p_t = p_features[:, T_in + t, :]  # [B, pressure_hidden_dim]

            force_t = self.force_head(
                pressure_feat=p_t,
                state=current_state,
                flow_context=flow_context,
                re_embed=re_embed,
            )  # [B, 2]

            state_ode = self.ode_solver.step(current_state, force_t)  # [B, 4]

            next_state = self.state_correction(state_ode, force_t, p_t)  # [B, 4]

            all_forces.append(force_t)
            all_states.append(next_state)

            if use_tf and t < T_out - 1:
                use_true = torch.rand(1).item() < teacher_forcing_ratio
                if use_true:
                    current_state = true_states[:, t, :]
                else:
                    current_state = next_state
            else:
                current_state = next_state

        forces = torch.stack(all_forces, dim=1)  # [B, T_out, 2]
        states = torch.stack(all_states, dim=1)  # [B, T_out, 4]

        return {
            'forces': forces,
            'states': states,
        }

# Loss Functions
def compute_viv_loss(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    weights: Dict[str, float] = None,
    epoch: int = 0,
) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = {
            'force': 1.0,
            'position': 5.0,
            'velocity': 2.0,
            'variance': 0.2,
            'smoothness': 0.02,
        }

    forces_pred = pred['forces']        # [B, T, 2]
    states_pred = pred['states']        # [B, T, 4]
    forces_true = batch['force_output'].to(device)    # [B, T, 2]
    states_true = batch['state_output'].to(device)    # [B, T, 4]

    T = forces_pred.shape[1]

    # ---- 1. Force Loss ----
    force_x_loss = torch.abs(forces_pred[:, :, 0] - forces_true[:, :, 0]).mean()
    force_y_loss = torch.abs(forces_pred[:, :, 1] - forces_true[:, :, 1]).mean()

    force_loss = 0.3 * force_x_loss + 0.7 * force_y_loss

    # ---- 2. Position Loss ----
    pos_pred = states_pred[:, :, :2]    # [B, T, 2]
    pos_true = states_true[:, :, :2]
    pos_x_loss = torch.abs(pos_pred[:, :, 0] - pos_true[:, :, 0]).mean()
    pos_y_loss = torch.abs(pos_pred[:, :, 1] - pos_true[:, :, 1]).mean()
    pos_loss = 0.3 * pos_x_loss + 0.7 * pos_y_loss

    # ---- 3. Velocity Loss ----
    vel_pred = states_pred[:, :, 2:]    # [B, T, 2]
    vel_true = states_true[:, :, 2:]
    vel_loss = torch.abs(vel_pred - vel_true).mean()

    # std over batch×time
    pred_std_fx = torch.std(forces_pred[:, :, 0])
    pred_std_fy = torch.std(forces_pred[:, :, 1])
    true_std_fx = torch.std(forces_true[:, :, 0]).detach()
    true_std_fy = torch.std(forces_true[:, :, 1]).detach()

    eps = 1e-6
    var_loss_fx = torch.abs(torch.log(pred_std_fx + eps) - torch.log(true_std_fx + eps))
    var_loss_fy = torch.abs(torch.log(pred_std_fy + eps) - torch.log(true_std_fy + eps))

    variance_loss = 0.3 * var_loss_fx + 0.7 * var_loss_fy

    pos_diff2 = torch.diff(torch.diff(pos_pred, dim=1), dim=1)
    smoothness_loss = torch.mean(pos_diff2 ** 2)

    total = (
        weights['force'] * force_loss +
        weights['position'] * pos_loss +
        weights['velocity'] * vel_loss +
        weights['variance'] * variance_loss +
        weights['smoothness'] * smoothness_loss
    )

    return {
        'total': total,
        'force': force_loss,
        'force_x': force_x_loss,
        'force_y': force_y_loss,
        'position': pos_loss,
        'velocity': vel_loss,
        'variance': variance_loss,
        'smoothness': smoothness_loss,
    }
