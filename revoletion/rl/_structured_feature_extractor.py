import logging

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ._features import (
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_EFUS_AVAILABILITY_FORECAST,
    OBS_KEY_EFUS_AVAILABLE_NOW,
    OBS_KEY_EFUS_CURRENT_SOC_DIFF,
    OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
    OBS_KEY_EFUS_REQUIRED_SOCS_FORECAST,
    OBS_KEY_EFUS_SOC,
    OBS_KEY_EFUS_URGENCY,
    OBS_KEY_FIXED_DEMANDS,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_GRID_EXPORT_COSTS,
    OBS_KEY_GRID_IMPORT_COSTS,
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

_VEHICLE_FEATURES = [
    OBS_KEY_EFUS_SOC,
    OBS_KEY_EFUS_CURRENT_SOC_DIFF,
    OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF,
    OBS_KEY_EFUS_AVAILABLE_NOW,
    OBS_KEY_EFUS_URGENCY,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
]


_FLEET_FEATURES = _VEHICLE_FEATURES + [OBS_KEY_FLEETS_IN_POWER, OBS_KEY_FLEETS_OUT_POWER]

_FORECAST_FEATURES = [
    OBS_KEY_EFUS_REQUIRED_SOCS_FORECAST,
    OBS_KEY_EFUS_AVAILABILITY_FORECAST,
    OBS_KEY_RENEWABLES_SCHEDULE,
    OBS_KEY_GRID_EXPORT_COSTS,
    OBS_KEY_GRID_IMPORT_COSTS,
]


class ForecastEncoder(nn.Module):
    def __init__(self, embed_dim: int = 8, output_dim: int = 4) -> None:
        super().__init__()
        self.conv0 = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=3,
        )
        self.activation0 = nn.ReLU()
        self.proj0 = nn.Linear(embed_dim, output_dim)

    def forward(self, x) -> torch.Tensor:
        b, n, t = x.shape
        x = x.view(b * n, 1, t)
        x = self.conv0(x)
        x = self.activation0(x)
        x = x.mean(dim=-1)
        x = self.proj0(x)
        x = x.view(b, n * x.shape[-1])
        return x


class StructuredFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embed_dim: int = 8,
        features_dim: int = 128,
    ):
        observation_space_dict = observation_space.spaces
        concat_dim = 0
        extractors = {}
        for key, space in observation_space_dict.items():
            if key in _FORECAST_FEATURES:
                n_entities = space.shape[0]
                dim = 4
                concat_dim += dim * n_entities
                extractors[key] = ForecastEncoder(embed_dim=embed_dim, output_dim=dim)
            else:
                dim = get_flattened_obs_dim(space)
                concat_dim += dim
                extractors[key] = nn.Flatten()

        n_vehicles = 0
        vehicle_dim = 0
        for key in _VEHICLE_FEATURES:
            feature_shape = observation_space_dict[key].shape
            if feature_shape is None:
                continue

            n_vehicles = feature_shape[0]
            if len(feature_shape) == 1:
                vehicle_dim += 1
            else:
                vehicle_dim += feature_shape[1]

        internal_dim = concat_dim + (n_vehicles * 3)

        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict(extractors)

        self.vehicle_encoder0 = nn.Sequential(
            nn.Linear(vehicle_dim, embed_dim * 2),
            nn.Tanh(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.LayerNorm(embed_dim),
        )

        self.fusion0 = nn.Sequential(
            nn.Linear(internal_dim, features_dim * 2),
            nn.Tanh(),
            nn.LayerNorm(features_dim * 2),
            nn.Linear(features_dim * 2, features_dim),
            nn.Tanh(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        vehicle_parts = []
        for key in _VEHICLE_FEATURES:
            part = observations[key]
            if part.dim() == 2:
                vehicle_parts.append(part.unsqueeze(-1))
            else:
                vehicle_parts.append(part)

        vehicle_parts_tensor = torch.cat(vehicle_parts, dim=-1)
        encoded_vehicle_parts = self.vehicle_encoder0(vehicle_parts_tensor)
        encoded_vehicle_parts_mean = encoded_vehicle_parts.mean(dim=-1)
        encoded_vehicle_parts_min, _ = encoded_vehicle_parts.min(dim=-1)
        encoded_vehicle_parts_max, _ = encoded_vehicle_parts.max(dim=-1)

        encoded_vehicle_statistics = torch.cat(
            [encoded_vehicle_parts_mean, encoded_vehicle_parts_min, encoded_vehicle_parts_max], dim=-1
        )

        global_parts = [encoded_vehicle_statistics]

        # Encode each key into one token
        for key, extractor in self.extractors.items():
            part = extractor(observations[key])
            global_parts.append(part)

        encoded_parts_tensor = torch.cat(global_parts, dim=1)

        fused_parts_tensor = self.fusion0(encoded_parts_tensor)
        return fused_parts_tensor
