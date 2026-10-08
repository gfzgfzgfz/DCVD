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
    root_contrastive_weight=0.5,
    level_contrastive_weight=0.5,
    level_weights=None,
):
    """计算根级、五层 CWE、跨模态对齐的联合损失。"""
    if len(level_logits) != 5:
        raise ValueError("Five hierarchy logits are required")
    if level_labels.ndim != 2 or level_labels.size(1) != 5:
        raise ValueError("level_labels must have shape [batch, 5]")
    if level_mask.shape != level_labels.shape:
        raise ValueError("level_mask must match level_labels")

    active_depth = max(1, min(int(active_depth), 5))
    if level_weights is None:
        level_weights = [1.0] * 5

    # 根级对全部样本计算：CE 学习判别边界，SupCon 分离 Safe/Vulnerable 表征。
    loss_root_ce = F.cross_entropy(root_logits, root_labels)
    loss_root_contrastive = contrastive_loss(representation, root_labels)

    loss_hierarchy_ce = representation.new_zeros(())
    loss_hierarchy_contrastive = representation.new_zeros(())

    # 训练阶段必须使用真实根标签门控：Safe 样本不参与后五层 CWE 损失。
    vulnerable = root_labels.eq(1)
    for level in range(active_depth):
        valid = vulnerable & level_mask[:, level].bool()
        if valid.any():
            weight = float(level_weights[level])
            # 按论文公式对已开放层级直接加权求和，不再除以有效层数。
            loss_hierarchy_ce = loss_hierarchy_ce + weight * F.cross_entropy(
                level_logits[level][valid], level_labels[valid, level]
            )
            loss_hierarchy_contrastive = (
                loss_hierarchy_contrastive
                + weight
                * contrastive_loss(
                    representation[valid], level_labels[valid, level]
                )
            )

    loss_root = loss_root_ce + root_contrastive_weight * loss_root_contrastive
    loss_hierarchy = (
        loss_hierarchy_ce
        + level_contrastive_weight * loss_hierarchy_contrastive
    )

    total = (
        root_weight * loss_root
        + hierarchy_weight * loss_hierarchy
        + cross_modal_weight * loss_cross_modal
    )
    components = {
        "root_ce": loss_root_ce.detach(),
        "root_contrastive": loss_root_contrastive.detach(),
        "hierarchy_ce_sum": loss_hierarchy_ce.detach(),
        "hierarchy_contrastive_sum": loss_hierarchy_contrastive.detach(),
        "cross_modal": loss_cross_modal.detach(),
        "root_total": loss_root.detach(),
        "hierarchy_total": loss_hierarchy.detach(),
    }
    return total, components
