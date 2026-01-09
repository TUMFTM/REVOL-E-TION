import logging

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
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
    OBS_KEY_EFUS_AVAILABLE,
    OBS_KEY_EFUS_REQUIRED_SOCS,
    OBS_KEY_EFUS_REAL_POWER_UNIT,
]

_FLEET_FEATURES = _VEHICLE_FEATURES + [OBS_KEY_FLEETS_IN_POWER, OBS_KEY_FLEETS_OUT_POWER]


class StructuredFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        embed_dim: int = 128,
        num_attention_heads: int = 4,
    ):
        observation_space_dict = observation_space.spaces
        concat_dim = 0
        extractors = {}
        for key, space in observation_space_dict.items():
            dim = get_flattened_obs_dim(space)
            concat_dim += dim
            extractors[key] = nn.Flatten()

        vehicle_dim = 0
        for key in _VEHICLE_FEATURES:
            feature_shape = observation_space_dict[key].shape
            if feature_shape is None:
                continue

            if len(feature_shape) == 1:
                vehicle_dim += 1
            else:
                vehicle_dim += feature_shape[1]

        features_dim = concat_dim + embed_dim

        super().__init__(observation_space, features_dim)

        self.extractors = nn.ModuleDict(extractors)

        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_dim, vehicle_dim * 2),
            nn.ReLU(),
            nn.Linear(vehicle_dim * 2, embed_dim),
            nn.ReLU(),
        )

        self.vehicle_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_parts = []

        # Encode each key into one token
        for key, extractor in self.extractors.items():
            part = extractor(observations[key])
            encoded_parts.append(part)

        vehicle_parts = []
        for key in _VEHICLE_FEATURES:
            part = observations[key]
            if part.dim() == 2:
                vehicle_parts.append(part.unsqueeze(-1))
            else:
                vehicle_parts.append(part)

        vehicle_parts_tensor = torch.cat(vehicle_parts, dim=-1)
        encoded_vehicle_parts = self.vehicle_encoder(vehicle_parts_tensor)

        vehicle_pooled = self.vehicle_attention(encoded_vehicle_parts, encoded_vehicle_parts, encoded_vehicle_parts)

        encoded_parts.append(vehicle_pooled)

        return torch.cat(encoded_parts, dim=1)
