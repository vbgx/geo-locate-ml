from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoSoftTargetLoss(nn.Module):
    """
    Distance-aware soft target loss:
    For each target y, define soft targets:
        p_j ∝ exp(-D[y,j] / tau)
    then minimize KL( p || softmax(logits) ).

    D: [C, C] distances (km), tau: km scale
    """
    def __init__(self, distance_km: torch.Tensor, tau_km: float):
        super().__init__()
        assert distance_km.dim() == 2 and distance_km.size(0) == distance_km.size(1)
        self.register_buffer("D", distance_km.float())
        self.tau = float(tau_km)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        B, C = logits.shape
        # gather distances for each target row: [B, C]
        Dy = self.D[targets]  # indexing on buffer
        # build soft targets
        p = torch.exp(-Dy / self.tau)
        p = p / (p.sum(dim=1, keepdim=True) + 1e-12)

        log_q = F.log_softmax(logits, dim=1)
        # KL(p || q) = sum p * (log p - log q) ; log p constant w.r.t logits, so minimize -sum p*log q
        loss = -(p * log_q).sum(dim=1).mean()
        return loss
