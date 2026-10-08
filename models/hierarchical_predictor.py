import torch.nn as nn


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
