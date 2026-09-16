import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskPredictor(nn.Module):
    """
    多任务监督学习
    """
    def __init__(self, k_dim, num_heads=8, alpha=0.5, beta=0.2):
        super(MultiTaskPredictor, self).__init__()
        
        self.alpha = alpha
        self.beta = beta

        # 函数级
        self.top_mlp = nn.Sequential(
            nn.Linear(k_dim, k_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(k_dim // 2, 1) 
        )

        # 代码级
        self.bottom_self_attn = nn.MultiheadAttention(embed_dim=k_dim, num_heads=num_heads, batch_first=True)
        self.bottom_mlp = nn.Sequential(
            nn.Linear(k_dim, k_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(k_dim * 2, k_dim)
        )
        self.bottom_norm = nn.LayerNorm(k_dim)
        
        # 将每个Token的特征向量聚合成一维标量
        self.token_to_scalar = nn.Linear(k_dim, 1)

    def forward(self, final_k, line_mask, is_training=True, y_f=None, y_s=None, L_ag=0):
        # 函数级
        pooled_k = final_k.mean(dim=1)
        logits_f = self.top_mlp(pooled_k).squeeze(-1) 
        prob_f = torch.sigmoid(logits_f)

        # 代码级
        attn_out, _ = self.bottom_self_attn(query=final_k, key=final_k, value=final_k)
        mlp_out = self.bottom_mlp(attn_out)
        norm_out = self.bottom_norm(final_k + mlp_out)
        
        # 给每个Token打分
        token_scalars = self.token_to_scalar(norm_out).squeeze(-1)
        
        # 利用掩码将Token分组到对应的行并求和
        line_sums = torch.bmm(line_mask.float(), token_scalars.unsqueeze(-1)).squeeze(-1)
        
        # 计算每行有几个Token 求平均
        line_counts = line_mask.sum(dim=-1).float()
        line_counts_clamped = torch.clamp(line_counts, min=1.0)
        line_features = line_sums / line_counts_clamped
        
        # 屏蔽掉没有代码的空白行
        invalid_lines = (line_counts == 0)
        line_features = line_features.masked_fill(invalid_lines, -1e9)
        
        # 转化为概率分布
        prob_s = torch.sigmoid(line_features)

        if not is_training:
            return prob_f, prob_s

        # 函数级
        loss_fn_f = nn.BCEWithLogitsLoss()
        L_f = loss_fn_f(logits_f, y_f.float())
        
        eps = 1e-9
        
        # 屏蔽掉 Padding 行
        valid_mask = (~invalid_lines).float()
        
        # y * log(y / p)
        term1 = y_s * torch.log((y_s + eps) / (prob_s + eps))
        # (1 - y) * log((1 - y) / (1 - p))
        term2 = (1 - y_s) * torch.log((1 - y_s + eps) / (1 - prob_s + eps))
        
        # 只保留有效行的损失
        kl_div_per_line = (term1 + term2) * valid_mask
        
        # 求所有有效行的损失和 除以 batch 大小)
        batch_size = line_features.size(0)
        L_s = torch.sum(kl_div_per_line) / batch_size
        
        # 总损失
        Loss = self.alpha * (L_f + self.beta * L_ag) + (1 - self.alpha) * L_s
        
        return prob_f, prob_s, Loss