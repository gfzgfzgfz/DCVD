import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerLLMModule(nn.Module):
    """
    真正对齐文档逻辑的 Transformer 提炼模块
    将全局融合特征 h_m 展开为序列矩阵，并赋予位置编码
    """
    def __init__(self, hidden_dim, k_dim, model_name="microsoft/graphcodebert-base"):
        super(TransformerLLMModule, self).__init__()
        # 预训练模型
        self.llm = AutoModel.from_pretrained(model_name)
        llm_hidden_size = self.llm.config.hidden_size
        
        # 冻结嵌入层和pooler的参数
        for param in self.llm.embeddings.parameters():
            param.requires_grad = False
        for param in self.llm.pooler.parameters():
            param.requires_grad = False

        # 维度投影
        self.align_projection = nn.Linear(hidden_dim, llm_hidden_size) if hidden_dim != llm_hidden_size else nn.Identity()
        
        # 输出维度投影
        self.k_projection = nn.Linear(llm_hidden_size, k_dim) if llm_hidden_size != k_dim else nn.Identity()

    def forward(self, h_m, attention_mask):
        """
        输入:
        h_m: (Batch, Seq_Len, hidden_dim)
        attention_mask: (Batch, Seq_Len) 序列掩码矩阵 真实为1 Padding为0
        """
        
        # 投影
        inputs_embeds = self.align_projection(h_m)
        
        # 调用LLM
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
        # 提取完整的隐状态序列
        # (Batch, Seq_Len, llm_hidden_size)
        sequence_features = outputs.last_hidden_state
        
        # 投影到最终的k_dim
        # (Batch, Seq_Len, k_dim)
        final_k = self.k_projection(sequence_features)
        
        return final_k