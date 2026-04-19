"""Network modules and the `Networks` container that holds H, G, F."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from efficientzero.envs.spec import EnvSpec
from efficientzero.networks.dynamics import DynamicsNet
from efficientzero.networks.prediction import PredictionNet
from efficientzero.networks.representation import RepresentationNet
from efficientzero.networks.shared import ProjectionHead
from efficientzero.utils.support import Support


@dataclass
class NetworkOutput:
    latent: torch.Tensor
    reward: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor


class Networks(nn.Module):
    """Container holding H (representation), G (dynamics), F (prediction).

    Also owns the SimSiam-style projection/prediction heads used by the
    self-supervised consistency loss (EfficientZero §3.4 / EZ-V2 §3.4).
    """

    def __init__(self, cfg: DictConfig, spec: EnvSpec):
        super().__init__()
        self.spec = spec
        net_cfg = cfg.network
        self.hidden_dim = int(net_cfg.hidden_dim)
        self.value_support = Support(
            size=int(net_cfg.value_support_size),
            vmin=float(net_cfg.value_support_min),
            vmax=float(net_cfg.value_support_max),
        )
        self.reward_support = Support(
            size=int(net_cfg.reward_support_size),
            vmin=float(net_cfg.reward_support_min),
            vmax=float(net_cfg.reward_support_max),
        )
        self.representation = RepresentationNet(spec, net_cfg)
        self.dynamics = DynamicsNet(spec, net_cfg, reward_support_size=self.reward_support.size)
        self.prediction = PredictionNet(spec, net_cfg, value_support_size=self.value_support.size)
        self.projection = ProjectionHead(self.hidden_dim, self.hidden_dim)
        self.predictor = ProjectionHead(self.hidden_dim, self.hidden_dim, predictor=True)

    def initial_inference(self, obs: torch.Tensor) -> NetworkOutput:
        latent = self.representation(obs)
        policy_logits, value_logits = self.prediction(latent)
        zero_reward = torch.zeros(
            latent.shape[0], self.reward_support.size, device=latent.device
        )
        return NetworkOutput(latent=latent, reward=zero_reward, policy_logits=policy_logits, value=value_logits)

    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        next_latent, reward_logits = self.dynamics(latent, action)
        policy_logits, value_logits = self.prediction(next_latent)
        return NetworkOutput(
            latent=next_latent, reward=reward_logits, policy_logits=policy_logits, value=value_logits
        )

    def project(self, latent: torch.Tensor, with_predictor: bool) -> torch.Tensor:
        z = self.projection(latent)
        if with_predictor:
            z = self.predictor(z)
        return z


def build_networks(cfg: DictConfig, spec: EnvSpec) -> Networks:
    return Networks(cfg, spec)


__all__ = ["Networks", "NetworkOutput", "build_networks"]
