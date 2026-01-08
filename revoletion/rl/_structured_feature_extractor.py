import logging

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ._features import (
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_EFUS_AVAILABLE,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
    OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_SOC,
    OBS_KEY_FIXED_DEMANDS,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_GRID_EXPORT_COSTS,
    OBS_KEY_GRID_EXPORT_POWER,
    OBS_KEY_GRID_IMPORT_COSTS,
    OBS_KEY_GRID_IMPORT_POWER,
    OBS_KEY_RENEWABLES_POWER,
    OBS_KEY_RENEWABLES_SCHEDULE,
    OBS_KEY_STATIONARY_BATTERIES_SOC,
    OBS_KEY_TIME_FEATURES,
)

_LOGGER = logging.getLogger(__name__)

_SCALAR_FEATURES = {
    OBS_KEY_TIME_FEATURES,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_RENEWABLES_POWER,
    OBS_KEY_STATIONARY_BATTERIES_SOC,
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_FIXED_DEMANDS,
}

_VEHICLE_ENCODER_FEATURES = {
    OBS_KEY_EFUS_SOC,
    OBS_KEY_EFUS_AVAILABLE,
    OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
}


class StructuredFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        forecast_horizon: int = 16,
        features_dim: int = 256,
        vehicle_embed_dim: int = 64,
        forecast_embed_dim: int = 32,
        use_attention: bool = True,
        num_attention_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        self._observation_space = observation_space.spaces
        self._forecast_horizon = forecast_horizon

        self._use_attention = use_attention
        self._vehicle_embed_dim = vehicle_embed_dim
        self._forecast_embed_dim = forecast_embed_dim

        # Each vehicle has: SoC (1) + real power unit + availability forecast (H) + required SoC forecast (H)
        vehicle_input_dim = 2 + 2 * self._forecast_horizon

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_input_dim, vehicle_embed_dim * 2),
            nn.LayerNorm(vehicle_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(vehicle_embed_dim * 2, vehicle_embed_dim),
            nn.LayerNorm(vehicle_embed_dim),
            nn.ReLU(),
        )

        # Attention-based pooling across vehicles
        if use_attention:
            self.vehicle_attention = nn.MultiheadAttention(
                embed_dim=vehicle_embed_dim,
                num_heads=num_attention_heads,
                batch_first=True,
                dropout=0.1,
            )
            # Learnable query token for pooling
            self.vehicle_query = nn.Parameter(torch.randn(1, 1, vehicle_embed_dim))

        vehicle_output_dim = vehicle_embed_dim

        # ============================================
        # 2. Renewable Generation Forecast Processing
        # ============================================
        if OBS_KEY_RENEWABLES_SCHEDULE in self._observation_space:
            self.n_renewables = self._observation_space[OBS_KEY_RENEWABLES_SCHEDULE].shape[0]

            # 1D CNN to extract temporal patterns from forecasts
            self.renewable_encoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.n_renewables,
                    out_channels=forecast_embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=forecast_embed_dim,
                    out_channels=forecast_embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
            )
            renewable_output_dim = forecast_embed_dim
        else:
            renewable_output_dim = 0

        # ============================================
        # 3. Grid Features Processing
        # ============================================
        if OBS_KEY_GRID_IMPORT_COSTS in self._observation_space:
            # import_costs, import_power, export_costs, export_power
            grid_input_dim = self._observation_space[OBS_KEY_GRID_IMPORT_COSTS].shape[0] * 4
            grid_output_dim = 32
            self.grid_encoder = nn.Sequential(
                nn.Linear(grid_input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, grid_output_dim),
                nn.ReLU(),
            )
        else:
            grid_output_dim = 0

        # ============================================
        # 4. Scalar Features Processing
        # ============================================
        scalar_dim = 0

        for scalar_feature in _SCALAR_FEATURES:
            if scalar_feature not in self._observation_space:
                continue
            scalar_dim += self._observation_space[scalar_feature].shape[0]

        if scalar_dim > 0:
            scalar_output_dim = 64
            self.scalar_encoder = nn.Sequential(
                nn.Linear(scalar_dim, 128),
                nn.ReLU(),
                nn.Linear(128, scalar_output_dim),
                nn.ReLU(),
            )
        else:
            scalar_output_dim = 0

        # ============================================
        # 5. Fusion Layer
        # ============================================
        total_dim = vehicle_output_dim + renewable_output_dim + grid_output_dim + scalar_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            observations: Dictionary of observations from the environment

        Returns:
            Encoded features of shape (batch_size, features_dim)
        """
        encoded_parts = []

        # Concatenate per-vehicle features
        vehicle_features = torch.cat(
            [
                observations[OBS_KEY_EFUS_SOC].unsqueeze(-1),  # (B, N, 1)
                observations[OBS_KEY_EFUS_REAL_POWER_UNIT].unsqueeze(-1),
                observations[OBS_KEY_EFUS_AVAILABLE],  # (B, N, H)
                observations[OBS_KEY_EFUS_REQUIRED_SOCS],  # (B, N, H)
            ],
            dim=-1,
        )  # (B, N, 1+2H)

        # Encode each vehicle
        B, N, F = vehicle_features.shape
        vehicle_features_flat = vehicle_features.view(B * N, F)
        vehicle_encoded = self.vehicle_encoder(vehicle_features_flat)
        vehicle_encoded = vehicle_encoded.view(B, N, self._vehicle_embed_dim)

        # Pool across vehicles
        if self._use_attention:
            # Use learnable query for attention pooling
            query = self.vehicle_query.expand(B, -1, -1)  # (B, 1, D)
            pooled_vehicle, _ = self.vehicle_attention(query, vehicle_encoded, vehicle_encoded)
            pooled_vehicle = pooled_vehicle.squeeze(1)  # (B, D)
        else:
            # Simple mean pooling
            pooled_vehicle = vehicle_encoded.mean(dim=1)  # (B, D)

        encoded_parts.append(pooled_vehicle)

        # ============================================
        # 2. Process Renewable Forecasts
        # ============================================
        if OBS_KEY_RENEWABLES_SCHEDULE in observations:
            renewable_forecast = observations[OBS_KEY_RENEWABLES_SCHEDULE]  # (B, N, H)
            renewable_encoded = self.renewable_encoder(renewable_forecast)  # (B, D, 1)
            renewable_encoded = renewable_encoded.squeeze(-1)  # (B, D)
            encoded_parts.append(renewable_encoded)

        # ============================================
        # 3. Process Grid Features
        # ============================================
        if OBS_KEY_GRID_IMPORT_COSTS in observations:
            grid_features = torch.cat(
                [
                    observations[OBS_KEY_GRID_IMPORT_COSTS],
                    observations[OBS_KEY_GRID_IMPORT_POWER],
                    observations[OBS_KEY_GRID_EXPORT_COSTS],
                    observations[OBS_KEY_GRID_EXPORT_POWER],
                ],
                dim=-1,
            )
            grid_encoded = self.grid_encoder(grid_features)
            encoded_parts.append(grid_encoded)

        # ============================================
        # 4. Process Scalar Features
        # ============================================
        scalar_features = []

        if OBS_KEY_TIME_FEATURES in observations:
            scalar_features.append(observations[OBS_KEY_TIME_FEATURES])
        if OBS_KEY_FLEETS_IN_POWER in observations:
            scalar_features.append(observations[OBS_KEY_FLEETS_IN_POWER])
            scalar_features.append(observations[OBS_KEY_FLEETS_OUT_POWER])
        if OBS_KEY_RENEWABLES_POWER in observations:
            scalar_features.append(observations[OBS_KEY_RENEWABLES_POWER])
        if OBS_KEY_STATIONARY_BATTERIES_SOC in observations:
            scalar_features.append(observations[OBS_KEY_STATIONARY_BATTERIES_SOC])
        if OBS_KEY_CONTROLLABLE_SOURCES_POWER in observations:
            scalar_features.append(observations[OBS_KEY_CONTROLLABLE_SOURCES_POWER])
        if OBS_KEY_FIXED_DEMANDS in observations:
            scalar_features.append(observations[OBS_KEY_FIXED_DEMANDS])

        if scalar_features:
            scalar_cat = torch.cat(scalar_features, dim=-1)
            scalar_encoded = self.scalar_encoder(scalar_cat)
            encoded_parts.append(scalar_encoded)

        combined = torch.cat(encoded_parts, dim=-1)
        output = self.fusion(combined)

        return output
