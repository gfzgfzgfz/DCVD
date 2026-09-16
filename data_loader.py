import os
import torch
import pandas as pd
from torch_geometric.loader import DataLoader

class VulDataLoader:
    """
    对齐基于PyG的图数据.pt和文本数据.csv
    """
    def __init__(self, pt_path, csv_path, batch_size=8, max_valid_rows=None, max_safe_lines=200, num_workers=0):
        self.pt_path = pt_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        # 代码读取上限 用于调试
        self.max_valid_rows = max_valid_rows
        # 安全行数上限 超过这个行数的代码会被过滤
        self.max_safe_lines = max_safe_lines
        self.num_workers = num_workers

    def _merge_graph_and_text_data(self):
        """
        内部方法：将图数据与 LLM 解释文本进行精确交集匹配
        """
        # CSV
        print("Loading code data...")
        df = pd.read_csv(self.csv_path)
        if self.max_valid_rows is not None:
            df = df.head(self.max_valid_rows)
        
        # 索引 -> (源代码, 解释文本) 的映射字典
        text_mapping = {}
        
        for _, row in df.iterrows():
            idx_val = int(row['index'])
            code_text = row['func_before']
            exp_text = row['llm_explanation']

            if len(code_text.strip()) > 0 and len(exp_text.strip()) > 0:
                text_mapping[idx_val] = (code_text, exp_text)

        # .pt
        print("Loading graph data...")
        raw_dataset = torch.load(self.pt_path, weights_only=False)
        
        merged_dataset = []
        # 被过滤的数据量
        filtered_count = 0
        for data in raw_dataset:
            graph_idx = data.index.item()
            
            # 如果这个图的索引在文本字典里 匹配成功
            if graph_idx in text_mapping:
                # 挂载文本
                data.code_text = text_mapping[graph_idx][0]
                data.exp_text = text_mapping[graph_idx][1]

                # 计算行数并过滤
                actual_lines = len(data.code_text.split('\n'))
                if actual_lines > self.max_safe_lines:
                    filtered_count += 1
                    continue
                
                # 创建全0张量
                dense_y_s = torch.zeros(self.max_safe_lines, dtype=torch.float)
                for flaw_idx in data.y_s.tolist():
                    if flaw_idx < self.max_safe_lines:
                        dense_y_s[flaw_idx] = 1.0
                            
                # 挂载为新的属性PyG DataLoader会将其自动堆叠为 [Batch, self.max_safe_lines]
                data.y_s_dense = dense_y_s
                merged_dataset.append(data)

        print("Filtered out {} samples exceeding {} lines.".format(filtered_count, self.max_safe_lines))
        return merged_dataset

    def get_dataloader(self, shuffle=True, drop_last=False):
        """
        获取PyTorch Geometric的DataLoader
        """
        dataset = self._merge_graph_and_text_data()
        
        if len(dataset) == 0:
            print("Warning: No valid data found after merging.")
            return None
            
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader