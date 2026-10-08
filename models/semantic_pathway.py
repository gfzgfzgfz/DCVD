import torch
import torch.nn as nn


class SemanticPathway(nn.Module):
    """使用共享预训练词嵌入表的语义通路，不调用完整 UniXcoder 编码器。"""

    def __init__(self, embedding_layer, hidden_size, out_features):
        super().__init__()
        self.shared_embedding = embedding_layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.ReLU(),
        )

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).to(
            device=token_embeddings.device, dtype=token_embeddings.dtype
        )
        summed = torch.sum(token_embeddings * expanded_mask, dim=1)
        count = expanded_mask.sum(dim=1).clamp(min=1.0)
        return summed / count

    def forward(self, code_ids, exp_ids, code_mask, exp_mask):
        # 源码与自然语言解释共享同一个 UniXcoder 词嵌入表。
        code_embeddings = self.shared_embedding(code_ids)
        explanation_embeddings = self.shared_embedding(exp_ids)

        explanation_global = self._mean_pooling(
            explanation_embeddings, exp_mask
        )
        # 将解释的全局语义注入每个源码 token，再映射到跨模态公共维度。
        code_features = self.projection(
            code_embeddings + explanation_global.unsqueeze(1)
        )

        code_global = self._mean_pooling(code_embeddings, code_mask)
        semantic_global = self.projection(code_global + explanation_global)
        return code_features, semantic_global
