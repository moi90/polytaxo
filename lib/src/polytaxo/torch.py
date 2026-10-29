from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def multilabel_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
):
    # Calculate the loss for each element individually
    loss = F.binary_cross_entropy_with_logits(
        input, target, weight=weight, reduction="none", pos_weight=pos_weight
    )

    # Apply focal loss if gamma is not zero
    if focal_gamma != 0.0:
        diff = torch.abs(torch.sigmoid(input) - target)  # 0 <= x <= 1
        loss = torch.pow(diff, focal_gamma) * loss

    # Filter out non-finite elements
    loss = torch.where(torch.isfinite(target), loss, 0)

    if reduction == "none":
        return loss

    if reduction == "mean":
        return loss.mean()

    if reduction == "sum":
        return loss.sum()

    raise ValueError(f"Unexpected reduction: {reduction}")


class MultiLabelLoss(nn.Module):
    """Modification of BCEWithLogitsLoss that works with integer targets (to support ignore_index)."""

    def __init__(
        self,
        reduction: str = "mean",
        focal_gamma=0.0,
    ) -> None:
        super().__init__()

        self.reduction = reduction
        self.focal_gamma = focal_gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return multilabel_loss(
            input,
            target,
            reduction=self.reduction,
            focal_gamma=self.focal_gamma,
        )


class BalancedMultiLabelLoss(MultiLabelLoss):
    """Balanced version of MultiLabelLoss"""

    count_pos: Optional[torch.Tensor]
    count_neg: Optional[torch.Tensor]

    def __init__(
        self,
        reduction: str = "mean",
        focal_gamma: float = 0.0,
        momentum: float = 0.9,
        balance_gamma: float = 0.5,
    ) -> None:
        super().__init__(reduction, focal_gamma)

        self.register_buffer("count_pos", None)
        self.register_buffer("count_neg", None)

        self.momentum = momentum
        self.balance_gamma = balance_gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            input.shape == target.shape
        ), f"Input and target shape do not match: {input.shape} != {target.shape}"

        # Estimate the number of positive and negative entries per channel
        with torch.no_grad():
            C = target.size(1)

            if target.ndim == 2:
                target_1d = target
            else:
                # Reshape target to (N,C):
                # (N, C, ...) => (N x ..., C)
                target_1d = target.moveaxis(1, -1).reshape((-1, C))

            batch_count_pos = (0.5 <= target_1d).sum(0)
            batch_count_neg = (target_1d < 0.5).sum(0)

            assert batch_count_pos.shape == (
                C,
            ), f"Unexpected shape: {batch_count_pos.shape}"
            assert batch_count_neg.shape == (
                C,
            ), f"Unexpected shape: {batch_count_neg.shape}"

            if self.count_pos is None or self.count_neg is None:
                self.count_pos = batch_count_pos
                self.count_neg = batch_count_neg
            else:
                self.count_pos = (self.momentum * self.count_pos) + (
                    (1 - self.momentum) * batch_count_pos
                )
                self.count_neg = (self.momentum * self.count_neg) + (
                    (1 - self.momentum) * batch_count_neg
                )

            pos_weight = torch.pow(
                self.count_neg / (self.count_pos + 1e-6), self.balance_gamma
            )

            missing_dims = max(0, target.ndim - 2)
            if missing_dims:
                # Reshape pos_weight so that it matches target
                # (C) => (C,1,...)
                pos_weight = pos_weight[(...,) + (None,) * missing_dims]

        return multilabel_loss(
            input,
            target,
            reduction=self.reduction,
            pos_weight=pos_weight,
            focal_gamma=self.focal_gamma,
        )
