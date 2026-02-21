from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    Resolution-agnostic CNN using AdaptiveAvgPool2d so 64/128/192 all work.

    Optionally predicts proxy attributes (multi-task):
      proxies: [p_water, p_urban, p_coastal, p_high_pop, p_high_elev] in [0,1]
    """
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.30,
        num_proxies: int = 0,
        *,
        num_classes_r6: int = 0,
        hierarchical_enabled: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_proxies = int(num_proxies)
        self.hierarchical_enabled = bool(hierarchical_enabled)
        self.num_classes_r6 = int(num_classes_r6) if hierarchical_enabled else 0

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # r7 (final) head
        self.fc = nn.Linear(128, self.num_classes)

        # r6 (coarse) head (optional)
        if self.hierarchical_enabled:
            if self.num_classes_r6 <= 0:
                raise ValueError("hierarchical_enabled=True requires num_classes_r6 > 0")
            self.fc_r6 = nn.Linear(128, self.num_classes_r6)
        else:
            self.fc_r6 = None

        # proxy head (optional)
        if self.num_proxies > 0:
            self.proxy_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, self.num_proxies),
            )
        else:
            self.proxy_head = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x  # [N,128]

    def forward(self, x: torch.Tensor):
        z = self._encode(x)
        logits_r7 = self.fc(z)

        if self.hierarchical_enabled:
            logits_r6 = self.fc_r6(z) if self.fc_r6 is not None else None
            if logits_r6 is None:
                raise RuntimeError("hierarchical_enabled=True but fc_r6 is None")
            if self.proxy_head is None:
                return logits_r6, logits_r7
            proxy_logits = self.proxy_head(z)
            proxy = torch.sigmoid(proxy_logits)  # [N,P] in [0,1]
            return logits_r6, logits_r7, proxy

        if self.proxy_head is None:
            return logits_r7
        proxy_logits = self.proxy_head(z)
        proxy = torch.sigmoid(proxy_logits)  # [N,P] in [0,1]
        return logits_r7, proxy
