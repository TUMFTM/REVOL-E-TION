import logging

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ._features import (
    OBS_KEY_CONTROLLABLE_SOURCES_POWER,
    OBS_KEY_EFUS_AVAILABLE_NOW,
    OBS_KEY_EFUS_CURRENT_SOC_DIFF,
    OBS_KEY_EFUS_NEXT_REQUIRED_SOC_DIFF,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
    OBS_KEY_EFUS_SOC,
    OBS_KEY_EFUS_URGENCY,
    OBS_KEY_FIXED_DEMANDS,
    OBS_KEY_FLEETS_IN_POWER,
    OBS_KEY_FLEETS_OUT_POWER,
    OBS_KEY_RENEWABLES_POWER,
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
    # OBS_KEY_EFUS_AVAILABLE,
    # OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
]

_FLEET_FEATURES = _VEHICLE_FEATURES + [OBS_KEY_FLEETS_IN_POWER, OBS_KEY_FLEETS_OUT_POWER]


class StructuredFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embed_dim: int = 8,
    ):
        observation_space_dict = observation_space.spaces
        concat_dim = 0
        extractors = {}
        for key, space in observation_space_dict.items():
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

        features_dim = concat_dim + (n_vehicles * 3)

        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict(extractors)

        self.vehicle_encoder0 = nn.Sequential(
            nn.Linear(vehicle_dim, embed_dim * 2),
            nn.Tanh(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
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

        return encoded_parts_tensor
