import torch
import torch.nn.functional as F


def compute_hierarchical_objective(
    representation,
    root_logits,
    level_logits,
    root_labels,
    level_labels,
    level_mask,
    contrastive_loss,
    loss_cross_modal,
    active_depth=5,
    root_weight=1.0,
    hierarchy_weight=1.0,
    cross_modal_weight=0.1,
    contrastive_weight=0.3,
    level_weights=None,
):
    """计算根级 CE、层次 CE、跨模态损失和层次监督对比损失。"""
    if len(level_logits) != 5:
        raise ValueError("Five hierarchy logits are required")
    if level_labels.ndim != 2 or level_labels.size(1) != 5:
        raise ValueError("level_labels must have shape [batch, 5]")
    if level_mask.shape != level_labels.shape:
        raise ValueError("level_mask must match level_labels")

    active_depth = max(1, min(int(active_depth), 5))
    if level_weights is None:
        level_weights = [1.0] * 5

    loss_root = F.cross_entropy(root_logits, root_labels)
    loss_hierarchy = representation.new_zeros(())
    loss_hierarchical_contrastive = representation.new_zeros(())
    active_weight = 0.0

    # 安全样本只参与根级判别，后续 CWE 层级通过 valid mask 排除。
    vulnerable = root_labels.eq(1)
    for level in range(active_depth):
        valid = vulnerable & level_mask[:, level].bool()
        if valid.any():
            weight = float(level_weights[level])
            loss_hierarchy = loss_hierarchy + weight * F.cross_entropy(
                level_logits[level][valid], level_labels[valid, level]
            )
            loss_hierarchical_contrastive = (
                loss_hierarchical_contrastive
                + weight
                * contrastive_loss(
                    representation[valid], level_labels[valid, level]
                )
            )
            active_weight += weight

    if active_weight > 0:
        loss_hierarchy = loss_hierarchy / active_weight
        loss_hierarchical_contrastive = (
            loss_hierarchical_contrastive / active_weight
        )

    total = (
        root_weight * loss_root
        + hierarchy_weight * loss_hierarchy
        + cross_modal_weight * loss_cross_modal
        + contrastive_weight * loss_hierarchical_contrastive
    )
    components = {
        "root": loss_root.detach(),
        "hierarchy": loss_hierarchy.detach(),
        "cross_modal": loss_cross_modal.detach(),
        "hierarchical_contrastive": loss_hierarchical_contrastive.detach(),
    }
    return total, components
