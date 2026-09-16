import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusionModule(nn.Module):
    """
    特征融合模块
    双向交叉注意力机制和对比学习损失
    """
    def __init__(self, hidden_dim, tau=0.1, alpha=0.2):
        super(FeatureFusionModule, self).__init__()
        self.hidden_dim = hidden_dim
        # 温度系数 tau
        self.tau = tau

        self.W_Q_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_G = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.W_Q_G = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 融合映射矩阵Wm与激活函数
        self.W_m = nn.Linear(2 * hidden_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def compute_contrastive_loss(self, F_G_global, F_C_global):
        F_G_norm = F.normalize(F_G_global, p=2, dim=-1)
        F_C_norm = F.normalize(F_C_global, p=2, dim=-1)
        sim_matrix = torch.matmul(F_G_norm, F_C_norm.transpose(0, 1)) / self.tau
        labels = torch.arange(F_G_global.size(0), device=F_G_global.device)
        return F.cross_entropy(sim_matrix, labels)

    def forward(self, F_G_seq, F_C, graph_mask=None, F_C_global=None, is_training=True):
        """
        F_G_seq: (Batch, Nodes, hidden_dim)
        F_C: (Batch, Seq_Len, hidden_dim)
        graph_mask: (Batch, Nodes)  真实节点为True, padding为False
        F_C_global: (Batch, hidden_dim)
        """
        seq_len = F_C.size(1)

        # === (C 查 G) ===
        # token序列去查询图节点序列
        Q_C = self.W_Q_C(F_C)         # (B, Seq, D)
        K_G = self.W_K_G(F_G_seq)     # (B, Nodes, D)
        V_G = self.W_V_G(F_G_seq)     # (B, Nodes, D)

        # (B, Seq, D) x (B, D, Nodes) -> (B, Seq, Nodes)
        score_C_G = torch.matmul(Q_C, K_G.transpose(1, 2)) / (self.hidden_dim ** 0.5)

        # graph_mask: (B, Nodes) -> (B, 1, Nodes)
        expanded_graph_mask = graph_mask.unsqueeze(1)
        score_C_G = score_C_G.masked_fill(~expanded_graph_mask, -1e9)

        attn_C_G = F.softmax(score_C_G, dim=-1)   # 在 Nodes 维度归一化
        h_C_G = torch.matmul(attn_C_G, V_G)       # (B, Seq, D)

        # === (G 查 C) ===
        # 图节点序列去查询token序列
        Q_G = self.W_Q_G(F_G_seq)     # (B, Nodes, D)
        K_C = self.W_K_C(F_C)         # (B, Seq, D)
        V_C = self.W_V_C(F_C)         # (B, Seq, D)

        # (B, Nodes, D) x (B, D, Seq) -> (B, Nodes, Seq)
        score_G_C = torch.matmul(Q_G, K_C.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_G_C = F.softmax(score_G_C, dim=-1)   # 在 Seq 维度归一化

        h_G_C = torch.matmul(attn_G_C, V_C)       # (B, Nodes, D)

        # === 为了和 token 序列对齐，先把图侧结果池化成一个全局向量 ===
        mask_float = graph_mask.unsqueeze(-1).float()   # (B, Nodes, 1)
        h_G_C_sum = torch.sum(h_G_C * mask_float, dim=1)   # (B, D)
        h_G_C_count = torch.clamp(mask_float.sum(dim=1), min=1e-6)  # (B, 1)
        h_G_C_global = h_G_C_sum / h_G_C_count           # (B, D)

        # 扩展到 token 维度，方便拼接
        h_G_C_expanded = h_G_C_global.unsqueeze(1).expand(-1, seq_len, -1)  # (B, Seq, D)

        # === 融合 ===
        concat_h = torch.cat([h_C_G, h_G_C_expanded], dim=-1)   # (B, Seq, 2D)
        h_m = self.leaky_relu(self.W_m(concat_h))               # (B, Seq, D)

        if is_training:
            mask_float = graph_mask.unsqueeze(-1).float()
            F_G_global = torch.sum(F_G_seq * mask_float, dim=1) / torch.clamp(mask_float.sum(dim=1), min=1e-6)

            L_ag = self.compute_contrastive_loss(F_G_global, F_C_global)
            return h_m, L_ag

        return h_m