import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    """对齐并融合图结构特征与代码 token 语义特征。"""

    def __init__(self, hidden_dim, tau=0.1, alpha=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau

        self.W_Q_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_G = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_Q_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_m = nn.Linear(2 * hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def compute_contrastive_loss(self, graph_global, code_global):
        """计算图到代码、代码到图两个方向的对称 InfoNCE 损失。"""
        graph_global = F.normalize(graph_global, p=2, dim=-1)
        code_global = F.normalize(code_global, p=2, dim=-1)
        logits = torch.matmul(graph_global, code_global.transpose(0, 1)) / self.tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_g2c = F.cross_entropy(logits, labels)
        loss_c2g = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_g2c + loss_c2g)

    def forward(
        self,
        F_G_seq,
        F_C,
        graph_mask,
        code_mask,
        F_C_global=None,
        is_training=True,
    ):
        if graph_mask is None or code_mask is None:
            raise ValueError("graph_mask and code_mask are required")
        if graph_mask.shape != F_G_seq.shape[:2]:
            raise ValueError(
                f"graph_mask shape {tuple(graph_mask.shape)} does not match "
                f"F_G_seq shape {tuple(F_G_seq.shape[:2])}"
            )
        if code_mask.shape != F_C.shape[:2]:
            raise ValueError(
                f"code_mask shape {tuple(code_mask.shape)} does not match "
                f"F_C shape {tuple(F_C.shape[:2])}"
            )

        graph_valid = graph_mask[:, None, :].to(
            device=F_G_seq.device, dtype=torch.bool
        )
        code_valid = code_mask[:, None, :].to(device=F_C.device, dtype=torch.bool)

        # 代码 token 查询图节点；graph_valid 屏蔽补齐的虚拟节点。
        query_code = self.W_Q_C(F_C)
        key_graph = self.W_K_G(F_G_seq)
        value_graph = self.W_V_G(F_G_seq)
        score_c2g = torch.matmul(query_code, key_graph.transpose(1, 2))
        score_c2g = score_c2g / (self.hidden_dim**0.5)
        score_c2g = score_c2g.masked_fill(
            ~graph_valid, torch.finfo(score_c2g.dtype).min
        )
        hidden_c2g = torch.matmul(F.softmax(score_c2g, dim=-1), value_graph)

        # 图节点查询代码 token；code_valid 保证 padding token 不接收注意力。
        query_graph = self.W_Q_G(F_G_seq)
        key_code = self.W_K_C(F_C)
        value_code = self.W_V_C(F_C)
        score_g2c = torch.matmul(query_graph, key_code.transpose(1, 2))
        score_g2c = score_g2c / (self.hidden_dim**0.5)
        score_g2c = score_g2c.masked_fill(
            ~code_valid, torch.finfo(score_g2c.dtype).min
        )
        hidden_g2c = torch.matmul(F.softmax(score_g2c, dim=-1), value_code)

        # 将节点级结果池化成图级向量，再广播到每个代码 token。
        graph_mask_float = graph_mask.unsqueeze(-1).to(
            device=hidden_g2c.device, dtype=hidden_g2c.dtype
        )
        graph_global = (hidden_g2c * graph_mask_float).sum(dim=1)
        graph_global = graph_global / graph_mask_float.sum(dim=1).clamp(min=1.0)
        graph_for_each_token = graph_global.unsqueeze(1).expand(-1, F_C.size(1), -1)

        # 融合结果保持 [B, Seq, D]，并显式清零代码 padding 位置。
        fused = torch.cat([hidden_c2g, graph_for_each_token], dim=-1)
        fused = self.leaky_relu(self.W_m(fused))
        fused = fused * code_mask.unsqueeze(-1).to(
            device=fused.device, dtype=fused.dtype
        )

        if not is_training:
            return fused
        if F_C_global is None:
            raise ValueError("F_C_global is required while training")

        # 使用融合前的结构全局向量与语义全局向量计算跨模态对齐损失。
        source_graph_mask = graph_mask.unsqueeze(-1).to(
            device=F_G_seq.device, dtype=F_G_seq.dtype
        )
        source_graph_global = (F_G_seq * source_graph_mask).sum(dim=1)
        source_graph_global = source_graph_global / source_graph_mask.sum(dim=1).clamp(
            min=1.0
        )
        loss_cross_modal = self.compute_contrastive_loss(
            source_graph_global, F_C_global
        )
        return fused, loss_cross_modal
