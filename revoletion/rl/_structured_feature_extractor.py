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

        self.vehicle_proj0 = nn.Linear(vehicle_dim, embed_dim)

        self.vehicle_encoder0 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

        self.vehicle_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.vehicle_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.fusion0 = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(features_dim * 2),
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
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
        vehicle_parts_tensor = self.vehicle_proj0(vehicle_parts_tensor)
        encoded_vehicle_parts = vehicle_parts_tensor + self.vehicle_encoder0(vehicle_parts_tensor)

        batch_size = vehicle_parts_tensor.shape[0]
        query = self.vehicle_query.expand(batch_size, -1, -1)
        vehicle_pooled, _ = self.vehicle_attention(query, encoded_vehicle_parts, encoded_vehicle_parts)
        vehicle_pooled = vehicle_pooled.squeeze(dim=1)

        global_parts = []

        # Encode each key into one token
        for key, extractor in self.extractors.items():
            part = extractor(observations[key])
            global_parts.append(part)

        global_parts.append(vehicle_pooled)

        encoded_parts_tensor = torch.cat(global_parts, dim=1)

        encoded_parts_tensor = encoded_parts_tensor + self.fusion0(encoded_parts_tensor)

        return encoded_parts_tensor
