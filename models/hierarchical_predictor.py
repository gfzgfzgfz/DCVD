import torch
import torch.nn as nn


def decode_hierarchical_predictions(
    root_logits,
    level_logits,
    invalid_label=-100,
):
    """使用根级预测进行推理门控，Safe 样本不输出后续 CWE 标签。"""
    if len(level_logits) != 5:
        raise ValueError("Exactly five CWE hierarchy logits are required")

    root_predictions = root_logits.argmax(dim=-1)
    predicted_vulnerable = root_predictions.eq(1)
    level_predictions = torch.full(
        (root_predictions.size(0), 5),
        fill_value=invalid_label,
        dtype=torch.long,
        device=root_predictions.device,
    )

    # 只有根级预测为 Vulnerable 的样本才保留五层 CWE 预测。
    if predicted_vulnerable.any():
        for level, logits in enumerate(level_logits):
            level_predictions[predicted_vulnerable, level] = logits[
                predicted_vulnerable
            ].argmax(dim=-1)

    return root_predictions, level_predictions, predicted_vulnerable


class HierarchicalPredictor(nn.Module):
    """根级 Safe/Vulnerable 门控，加五个 CWE 抽象层级分类头。"""

    def __init__(self, input_dim, level_num_classes, dropout=0.2):
        super().__init__()
        if len(level_num_classes) != 5:
            raise ValueError("Exactly five CWE hierarchy levels are required")

        self.shared = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 根级头对全部样本训练；五个层级头只对有效漏洞样本训练。
        self.root_head = nn.Linear(input_dim, 2)
        self.level_heads = nn.ModuleList(
            nn.Linear(input_dim, count) for count in level_num_classes
        )

    def forward(self, representation):
        hidden = self.shared(representation)
        root_logits = self.root_head(hidden)
        level_logits = [head(hidden) for head in self.level_heads]
        return root_logits, level_logits
