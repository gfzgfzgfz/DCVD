import torch
import torch.nn as nn

class SemanticPathway(nn.Module):
    """
    语义信息通路模块
    """
    def __init__(self, embedding_layer, hidden_size, out_features):
        """
        embedding_layer: 从LLM中剥离出来的嵌入层
        hidden_size: 嵌入层的输出维度
        out_features: 最终需要与F_G对齐的输出维度
        """
        super(SemanticPathway, self).__init__()
        self.shared_embedding = embedding_layer

        # 最后利用非线性映射将F_G和F_C映射到相同的维度
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, out_features),
            nn.ReLU()
        )

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        计算忽略Padding的平均池化
        """
        # 扩展 mask 的维度
        # (Batch, Seq_Len, 1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 真实Token的embedding累加
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # 真实Token的数量
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 平均
        return sum_embeddings / sum_mask

    def forward(self, code_ids, exp_ids, code_mask, exp_mask, is_training=True):
        """
        code_ids: 代码的 Token ID 矩阵, (Batch, Seq_len_code)
        exp_ids: 解释文本的 Token ID 矩阵, (Batch, Seq_len_exp)
        code_mask/exp_mask: 掩码 维度同上, (真实为1，Pad为0)
        is_training: 是否为训练模式
        """
        
        # (Batch, Seq_Len, hidden_size)
        code_embeds = self.shared_embedding(code_ids)
        exp_embeds = self.shared_embedding(exp_ids)

        exp_vec = self._mean_pooling(exp_embeds, exp_mask)

        F_C = self.projection(code_embeds + exp_vec.unsqueeze(1))
        
        if is_training:
            code_vec = self._mean_pooling(code_embeds, code_mask)
            F_C_global = self.projection(code_vec + exp_vec)
            return F_C, F_C_global
        
        return F_C