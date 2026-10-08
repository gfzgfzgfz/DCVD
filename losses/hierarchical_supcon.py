import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """单个 CWE 层级上的监督对比损失。"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, dimension]")
        if labels.ndim != 1 or labels.size(0) != features.size(0):
            raise ValueError("labels must have shape [batch]")
        if features.size(0) < 2:
            return features.new_zeros(())

        features = F.normalize(features, p=2, dim=-1)
        logits = torch.matmul(features, features.transpose(0, 1))
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        sample_count = features.size(0)
        # 排除样本自身；同层标签相同的其他样本才是正样本。
        self_mask = torch.eye(
            sample_count, device=features.device, dtype=torch.bool
        )
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        # 没有同类正样本的 anchor 不参与该批次的对比损失。
        valid_anchor = positive_mask.any(dim=1)
        if not valid_anchor.any():
            return features.new_zeros(())

        exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12)
        )
        positive_count = positive_mask.sum(dim=1).clamp(min=1)
        mean_positive_log_prob = (
            positive_mask.to(log_prob.dtype) * log_prob
        ).sum(dim=1) / positive_count
        return -mean_positive_log_prob[valid_anchor].mean()
