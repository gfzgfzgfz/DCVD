import torch
import pandas as pd
from torch_geometric.loader import DataLoader


LEVEL_COLUMNS = [f"level_{index}" for index in range(1, 6)]
MASK_COLUMNS = [f"level_mask_{index}" for index in range(1, 6)]


class VulDataLoader:
    """按 index 合并 PyG 图、源码文本、解释文本和五级 CWE 标签。"""

    def __init__(
        self,
        pt_path,
        csv_path,
        batch_size=8,
        max_valid_rows=None,
        num_workers=0,
    ):
        self.pt_path = pt_path
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.max_valid_rows = max_valid_rows
        self.num_workers = num_workers

    @staticmethod
    def _validate_columns(frame):
        required = {
            "index",
            "func_before",
            "llm_explanation",
            "root_label",
            *LEVEL_COLUMNS,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

    @staticmethod
    def _row_to_metadata(row):
        # 缺失层级统一编码为 -100，并通过 level_mask 排除损失计算。
        labels = []
        masks = []
        for level_column, mask_column in zip(LEVEL_COLUMNS, MASK_COLUMNS):
            raw_label = row[level_column]
            valid_label = not pd.isna(raw_label) and int(raw_label) >= 0
            labels.append(int(raw_label) if valid_label else -100)
            if mask_column in row.index:
                masks.append(bool(int(row[mask_column])))
            else:
                masks.append(valid_label)

        # Safe 样本没有合法 CWE 路径，只参与 root_label 的二分类。
        if int(row["root_label"]) == 0:
            labels = [-100] * 5
            masks = [False] * 5

        return {
            "code": str(row["func_before"]),
            "explanation": str(row["llm_explanation"]),
            "root_label": int(row["root_label"]),
            "level_labels": labels,
            "level_mask": masks,
            "cwe_id": str(row.get("cwe_id", "SAFE")),
            "pair_id": str(row.get("pair_id", "")),
            "is_fixed": int(row.get("is_fixed", 0)),
        }

    def build_dataset(self):
        frame = pd.read_csv(self.csv_path)
        if self.max_valid_rows is not None:
            frame = frame.head(self.max_valid_rows)
        self._validate_columns(frame)

        metadata = {
            int(row["index"]): self._row_to_metadata(row)
            for _, row in frame.iterrows()
        }
        # .pt 文件仅保存结构信息，文本和标签以 CSV 为准。
        graph_dataset = torch.load(self.pt_path, weights_only=False)

        merged = []
        for graph in graph_dataset:
            graph_index = int(graph.index.item())
            if graph_index not in metadata:
                continue
            item = metadata[graph_index]
            if not item["code"].strip() or not item["explanation"].strip():
                continue

            graph.code_text = item["code"]
            graph.exp_text = item["explanation"]
            graph.root_label = torch.tensor([item["root_label"]], dtype=torch.long)
            graph.level_labels = torch.tensor(
                [item["level_labels"]], dtype=torch.long
            )
            graph.level_mask = torch.tensor(
                [item["level_mask"]], dtype=torch.bool
            )
            graph.cwe_id = item["cwe_id"]
            graph.pair_id = item["pair_id"]
            graph.is_fixed = torch.tensor([item["is_fixed"]], dtype=torch.long)
            merged.append(graph)

        if not merged:
            raise ValueError("No graph samples matched the CSV rows")
        return merged

    def get_dataloader(self, shuffle=True, drop_last=False):
        return DataLoader(
            self.build_dataset(),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=True,
        )
