# 当前分支：CWE 层次多分类改造

本目录在 DCVD 双通道主干上加入了根级漏洞判别、五级 CWE 分类头和分层监督对比损失。
原始 `models/multi_task_predictor.py` 仅保留为 DCVD 行定位任务的参考，新训练入口不再调用它。

## 新增模块

- `models/vulnerability_encoder.py`：封装结构分支、语义分支、融合模块和 GraphCodeBERT，输出 256 维函数表征。
- `models/hierarchical_predictor.py`：根级 Safe/Vulnerable 分类头、五个 CWE 层级分类头，以及按根级预测阻断 Safe 样本的推理解码函数。
- `losses/hierarchical_supcon.py`：同层同类样本的监督对比损失。
- `losses/hierarchical_loss.py`：组合根级 CE/SupCon、五层 CWE CE/SupCon 求和及跨模态损失。训练时使用真实根标签门控，Safe 样本不参与后五层损失。
- `data_loader.py`：读取五级 CWE 标签，不再读取 DCVD 的行级定位标签。

当前损失形式为：

```text
L = root_weight × (L_root_CE + λ × L_root_SupCon)
  + hierarchy_weight × Σ_l α_l(L_level_l_CE + μ × L_level_l_SupCon)
  + cross_modal_weight × L_cm
```

其中 `λ` 对应 `--root_contrastive_weight`，`μ` 对应
`--level_contrastive_weight`，默认均为 `0.5`。推理时根级预测为 Safe 的样本，
五层 CWE 预测统一返回 `-100`，不再给出漏洞类型。

## 生成最小 demo 数据

先安装 `requirements.txt` 中的 PyTorch Geometric，再从工作区根目录执行：

```powershell
python ".\脚本\生成层次分类_demo数据.py"
```

默认输出到 `data/demo/`，包含训练/验证 CSV、PyG 图文件、节点词表和类别数配置。
该数据只用于联调，不能用于论文实验结果。

## Demo 训练命令

```powershell
python train.py `
  --train_pt_path ./data/demo/train_graphs.pt `
  --train_csv_path ./data/demo/train.csv `
  --valid_pt_path ./data/demo/valid_graphs.pt `
  --valid_csv_path ./data/demo/valid.csv `
  --vocab_path ./data/demo/node_vocab.json `
  --class_counts_path ./data/demo/class_counts.json `
  --batch_size 12 `
  --num_train_epochs 5 `
  --epochs_per_level 1
```

首次运行会从 Hugging Face 下载 UniXcoder 和 GraphCodeBERT。真实数据还需要额外完成 AST/CFG
提取、CWE 层次映射和 LLM 解释生成。

## 原始 DCVD 说明


![11](./figs/title.png)![image-20260510182333988](./figs/main.png)



```bash
project/
├── train.py
├── data_loader.py
├── README.md
├── requirements.txt
├── models/
│   ├── control_pathway.py
│   ├── semantic_pathway.py
│   ├── feature_fusion_module.py
│   ├── transformer_llm_module.py
│   └── multi_task_predictor.py
```

# Data and Model Sources

## Dataset

We use the LineVul dataset for vulnerability detection:

> Michael Fu and Chakkrit Tantithamthavorn, "LineVul: A Transformer-based Line-Level Vulnerability Prediction", MSR 2022.  
> Paper: https://conf.researchr.org/details/msr-2022/msr-2022-technical-papers/26/LineVul-A-Transformer-based-Line-Level-Vulnerability-Prediction  
> Repository: https://github.com/awsm-research/LineVul

We further preprocess the dataset into graph structures and align each sample with LLM-generated explanations.

## Models

We use the following pretrained models:

* OpenAI GPT-4o-mini  
  > OpenAI, "Hello GPT-4o", 2024.  
  > URL: https://openai.com/index/hello-gpt-4o/  
  
  * gpt-4o-mini-2024-07-18: used for generating LLM explanations.

* Qwen Models (Qwen3 series)  
  > Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388, 2025.  
  Paper: https://arxiv.org/abs/2505.09388  
  Repository: https://github.com/QwenLM/Qwen3  

  * Qwen3.5-4B: used as the semantic embedding backbone (embedding layer only).

* **GraphCodeBERT**  
  > Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow", ICLR 2021.   
  > Paper: https://arxiv.org/abs/2009.08366  
  > Repository: https://github.com/microsoft/CodeBERT  

## Licenses

This project uses publicly available datasets and pretrained models. Their licenses are listed below:

* **LineVul Dataset**  
  License: MIT License  
  Source: https://github.com/awsm-research/LineVul  
  
* **OpenAI GPT-4o-mini**  
  Usage is subject to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use).

* **Qwen Models (Qwen3.5-4B)**  
  License: Apache License 2.0  
  Source: https://github.com/QwenLM/Qwen3  

* **GraphCodeBERT**  
  License: MIT License  
  Source: https://github.com/microsoft/CodeBERT  

All assets are used in accordance with their respective licenses.

# Environment Setup

## Requirements

```bash
python >= 3.9

torch>=2.0.0
torch-geometric>=2.4.0

transformers>=4.35.0
accelerate>=0.25.0
sentencepiece

pandas>=1.5.0
tqdm>=4.60.0

tensorboard
```
## PyTorch Geometric Installation
PyTorch Geometric (PyG) depends on CUDA. Please install it according to your environment.

# Data Preparation

## Directory Structure

```bash
data/
├── processed/
│   ├── train_data_masked.pt
│   ├── train_with_llm.csv
│   ├── val_data_masked.pt
│   ├── val_with_llm.csv
│   ├── test_data_masked.pt
│   ├── test_with_llm.csv
├── config/
│   └── node_vocab.json
```

## CSV Format

The CSV file must contain the following fields:

* `index`: unique sample ID
* `func_before`: source code
* `llm_explanation`: LLM-generated explanation

## Graph Data (.pt)

Each PyG Data object must include:

* `index`: aligned with CSV
* `x`: node features
* `ast_edge_index`: AST edges
* `cfg_edge_index`: CFG edges
* `y_f`: function-level label
* `y_s`: line-level labels
* `line_mask`: token-to-line mapping

# Model Configuration

The framework uses the following pretrained models:

* Semantic Pathway: `Qwen/Qwen3.5-4B`
* Transformer Module: `microsoft/graphcodebert-base`

You can modify them via:

```bash
--llm_model_name
--transformer_base_name
```

# Quickstart

```bash
python train.py \
  --train_pt_path ./data/processed/train_data_masked.pt \
  --train_csv_path ./data/processed/train_with_llm.csv \
  --valid_pt_path ./data/processed/val_data_masked.pt \
  --valid_csv_path ./data/processed/val_with_llm.csv \
  --vocab_path ./data/config/node_vocab.json \
  --output_dir ./outputs \
  --run_name demo
```

# Output

Results will be saved to:

```bash
outputs/<run_name>/
```

Including:

* model checkpoints
* logs
* result.json
