import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

class GATModel(nn.Module):
    """
    基于 PyG 的稀疏图 GAT 特征提取模型
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(GATModel, self).__init__()
        
        # 使用 PyG 官方封装的 GATConv
        self.layer1 = GATConv(in_features, hidden_features)
        self.layer2 = GATConv(hidden_features, out_features)

    def forward(self, x, edge_index):
        """
        x: 节点的初始特征向量矩阵, (全Batch节点总数, in_features)
        edge_index: 稀疏边索引, (2, 全Batch边总数)
        """
        # 两层GAT
        x = self.layer1(x, edge_index)
        x = F.elu(x)
        x = self.layer2(x, edge_index)
        x = F.elu(x)
        
        return x

class ControlPathway(nn.Module):
    """
    控制依赖通路模块
    """
    def __init__(self, vocab_size, embedding_dim, hidden_features, out_features):
        super(ControlPathway, self).__init__()
        # 嵌入层
        self.node_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # 分别处理 AST 和 CFG
        self.ast_gat = GATModel(embedding_dim, hidden_features, out_features)
        self.cfg_gat = GATModel(embedding_dim, hidden_features, out_features)

    def forward(self, x, ast_edge_index, cfg_edge_index, batch):
        """
        分别输入 AST 和 CFG 经过 PyG 封装后的稀疏数据
        """
        embedded_x = self.node_embedding(x)
        ast_node_feats = self.ast_gat(embedded_x, ast_edge_index)
        cfg_node_feats = self.cfg_gat(embedded_x, cfg_edge_index)
        
        # 逐元素相加
        fused_node_feats = ast_node_feats + cfg_node_feats

        F_G_seq, graph_mask = to_dense_batch(fused_node_feats, batch)
        
        return F_G_seq, graph_mask