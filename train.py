import argparse
import json
import logging
import math
import os

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

from data_loader import VulDataLoader
from losses import SupervisedContrastiveLoss, compute_hierarchical_objective
from models.control_pathway import ControlPathway
from models.feature_fusion_module import FeatureFusionModule
from models.hierarchical_predictor import (
    HierarchicalPredictor,
    decode_hierarchical_predictions,
)
from models.semantic_pathway import SemanticPathway
from models.transformer_llm_module import TransformerLLMModule
from models.vulnerability_encoder import VulnerabilityEncoder


LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DCVD backbone with hierarchical CWE contrastive learning"
    )

    # 数据文件：图数据负责 AST/CFG，CSV 负责源码、解释和层次标签。
    parser.add_argument("--train_pt_path", required=True)
    parser.add_argument("--train_csv_path", required=True)
    parser.add_argument("--valid_pt_path", required=True)
    parser.add_argument("--valid_csv_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--class_counts_path", required=True)

    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--run_name", default="hierarchical_cwe_demo")
    parser.add_argument("--embedding_model_name", default="microsoft/unixcoder-base")
    parser.add_argument(
        "--transformer_base_name", default="microsoft/graphcodebert-base"
    )
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--f_out_dim", type=int, default=128)
    parser.add_argument("--graph_embed_dim", type=int, default=128)
    parser.add_argument("--graph_hidden_dim", type=int, default=256)
    parser.add_argument("--k_dim", type=int, default=256)
    parser.add_argument("--representation_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--main_learning_rate", type=float, default=1e-4)
    parser.add_argument("--embedding_learning_rate", type=float, default=1e-6)
    parser.add_argument("--transformer_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--freeze_embedding", action="store_true")

    # 每经过若干轮开放一个更细的 CWE 层级，形成由粗到细的课程学习。
    parser.add_argument("--epochs_per_level", type=int, default=10)
    parser.add_argument("--cross_modal_temperature", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.07)
    parser.add_argument("--root_weight", type=float, default=1.0)
    parser.add_argument("--hierarchy_weight", type=float, default=1.0)
    parser.add_argument("--cross_modal_weight", type=float, default=0.1)
    # 与论文中的 lambda、mu 对应：分别控制根级与 CWE 层级 SupCon。
    parser.add_argument("--root_contrastive_weight", type=float, default=0.5)
    parser.add_argument("--level_contrastive_weight", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision", choices=["no", "fp16", "bf16"], default="no"
    )
    parser.add_argument("--report_to", default="none")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    return args


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def build_models(args, graph_vocab_size, level_num_classes):
    # UniXcoder 在本方案中只提供预训练词嵌入，深层上下文建模仍由 GraphCodeBERT 完成。
    embedding_backbone = AutoModel.from_pretrained(args.embedding_model_name)
    embedding_layer = embedding_backbone.get_input_embeddings().to(torch.float32)
    embedding_hidden_size = embedding_backbone.config.hidden_size

    semantic_model = SemanticPathway(
        embedding_layer=embedding_layer,
        hidden_size=embedding_hidden_size,
        out_features=args.f_out_dim,
    )
    semantic_model.shared_embedding.weight.requires_grad_(not args.freeze_embedding)
    del embedding_backbone

    control_model = ControlPathway(
        vocab_size=graph_vocab_size,
        embedding_dim=args.graph_embed_dim,
        hidden_features=args.graph_hidden_dim,
        out_features=args.f_out_dim,
    )
    fusion_model = FeatureFusionModule(
        hidden_dim=args.f_out_dim,
        tau=args.cross_modal_temperature,
    )
    transformer_model = TransformerLLMModule(
        hidden_dim=args.f_out_dim,
        k_dim=args.k_dim,
        model_name=args.transformer_base_name,
    )
    encoder = VulnerabilityEncoder(
        control_model=control_model,
        semantic_model=semantic_model,
        fusion_model=fusion_model,
        transformer_model=transformer_model,
        k_dim=args.k_dim,
        output_dim=args.representation_dim,
    )
    predictor = HierarchicalPredictor(
        input_dim=args.representation_dim,
        level_num_classes=level_num_classes,
    )
    return encoder, predictor


def build_optimizer(args, encoder, predictor):
    # 预训练词嵌入和 GraphCodeBERT 使用较小学习率，新建模块使用主学习率。
    embedding_parameters = list(
        encoder.semantic_model.shared_embedding.parameters()
    )
    transformer_parameters = list(encoder.transformer_model.parameters())
    special_ids = {
        id(parameter)
        for parameter in embedding_parameters + transformer_parameters
    }
    main_parameters = [
        parameter
        for parameter in list(encoder.parameters()) + list(predictor.parameters())
        if id(parameter) not in special_ids and parameter.requires_grad
    ]

    parameter_groups = [
        {
            "params": main_parameters,
            "lr": args.main_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for p in transformer_parameters if p.requires_grad],
            "lr": args.transformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]
    trainable_embedding = [p for p in embedding_parameters if p.requires_grad]
    if trainable_embedding:
        parameter_groups.append(
            {
                "params": trainable_embedding,
                "lr": args.embedding_learning_rate,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(parameter_groups)


def tokenize_batch(tokenizer, batch, max_length, device):
    code_tokens = tokenizer(
        list(batch.code_text),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    explanation_tokens = tokenizer(
        list(batch.exp_text),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return code_tokens, explanation_tokens


def unpack_labels(batch):
    # PyG 将每个样本的 [1, 5] 标签拼成 [B, 5]。
    root_labels = batch.root_label.view(-1).long()
    level_labels = batch.level_labels.view(-1, 5).long()
    level_mask = batch.level_mask.view(-1, 5).bool()
    return root_labels, level_labels, level_mask


def macro_f1(predictions, labels, class_count):
    scores = []
    for class_index in range(class_count):
        predicted = predictions.eq(class_index)
        actual = labels.eq(class_index)
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        if actual.any():
            scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


@torch.no_grad()
def evaluate(
    encoder,
    predictor,
    dataloader,
    tokenizer,
    accelerator,
    args,
    level_num_classes,
):
    encoder.eval()
    predictor.eval()
    root_predictions = []
    root_targets = []
    level_predictions = [[] for _ in range(5)]
    level_targets = [[] for _ in range(5)]

    for batch in dataloader:
        batch = batch.to(accelerator.device)
        code_tokens, explanation_tokens = tokenize_batch(
            tokenizer, batch, args.max_seq_len, accelerator.device
        )
        representation, _ = encoder(batch, code_tokens, explanation_tokens)
        root_logits, hierarchy_logits = predictor(representation)
        root_labels, hierarchy_labels, hierarchy_mask = unpack_labels(batch)

        # 推理/验证使用预测根标签门控；预测为 Safe 时五层结果均为 -100。
        root_batch_predictions, gated_level_predictions, _ = (
            decode_hierarchical_predictions(root_logits, hierarchy_logits)
        )

        gathered_root_predictions, gathered_root_labels = accelerator.gather_for_metrics(
            (root_batch_predictions, root_labels)
        )
        root_predictions.append(gathered_root_predictions.cpu())
        root_targets.append(gathered_root_labels.cpu())

        for level in range(5):
            # 真实标签只用于确定哪些样本具备 CWE 评估目标；预测门控由上方完成。
            # 若真实漏洞被根级误判为 Safe，其 -100 预测会在 CWE 指标中计为错误。
            valid = root_labels.eq(1) & hierarchy_mask[:, level]
            if not valid.any():
                continue
            predictions, targets = accelerator.gather_for_metrics(
                (
                    gated_level_predictions[valid, level],
                    hierarchy_labels[valid, level],
                )
            )
            level_predictions[level].append(predictions.cpu())
            level_targets[level].append(targets.cpu())

    root_predictions = torch.cat(root_predictions)
    root_targets = torch.cat(root_targets)
    metrics = {
        "root_accuracy": root_predictions.eq(root_targets).float().mean().item()
    }
    for level in range(5):
        if not level_predictions[level]:
            metrics[f"level_{level + 1}_accuracy"] = 0.0
            metrics[f"level_{level + 1}_macro_f1"] = 0.0
            continue
        predictions = torch.cat(level_predictions[level])
        targets = torch.cat(level_targets[level])
        metrics[f"level_{level + 1}_accuracy"] = (
            predictions.eq(targets).float().mean().item()
        )
        metrics[f"level_{level + 1}_macro_f1"] = macro_f1(
            predictions, targets, level_num_classes[level]
        )

    metrics["selection_score"] = 0.5 * metrics["root_accuracy"] + 0.5 * metrics[
        "level_5_macro_f1"
    ]
    return metrics


def save_checkpoint(accelerator, encoder, predictor, args, metrics, filename):
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "encoder": accelerator.get_state_dict(encoder),
        "predictor": accelerator.get_state_dict(predictor),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, os.path.join(args.output_dir, filename))


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    graph_vocab_size = len(load_json(args.vocab_path))
    class_config = load_json(args.class_counts_path)
    level_num_classes = class_config["level_num_classes"]
    if len(level_num_classes) != 5 or min(level_num_classes) < 1:
        raise ValueError("level_num_classes must contain five positive integers")

    train_loader = VulDataLoader(
        args.train_pt_path,
        args.train_csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ).get_dataloader(shuffle=True, drop_last=True)
    valid_loader = VulDataLoader(
        args.valid_pt_path,
        args.valid_csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ).get_dataloader(shuffle=False, drop_last=False)

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    if tokenizer.pad_token is None:
        raise ValueError("The selected tokenizer has no padding token")

    encoder, predictor = build_models(
        args, graph_vocab_size, level_num_classes
    )
    optimizer = build_optimizer(args, encoder, predictor)
    encoder, predictor, optimizer, train_loader, valid_loader = accelerator.prepare(
        encoder, predictor, optimizer, train_loader, valid_loader
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * args.num_train_epochs
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    supcon = SupervisedContrastiveLoss(args.supcon_temperature)
    progress = tqdm(
        total=total_steps,
        disable=not accelerator.is_local_main_process,
        desc="training",
    )
    best_score = float("-inf")

    for epoch in range(args.num_train_epochs):
        encoder.train()
        predictor.train()
        active_depth = min(epoch // args.epochs_per_level + 1, 5)
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(accelerator.device)
            code_tokens, explanation_tokens = tokenize_batch(
                tokenizer, batch, args.max_seq_len, accelerator.device
            )
            root_labels, level_labels, level_mask = unpack_labels(batch)

            with accelerator.accumulate(encoder, predictor):
                representation, loss_cross_modal = encoder(
                    batch, code_tokens, explanation_tokens
                )
                root_logits, level_logits = predictor(representation)
                loss, components = compute_hierarchical_objective(
                    representation=representation,
                    root_logits=root_logits,
                    level_logits=level_logits,
                    root_labels=root_labels,
                    level_labels=level_labels,
                    level_mask=level_mask,
                    contrastive_loss=supcon,
                    loss_cross_modal=loss_cross_modal,
                    active_depth=active_depth,
                    root_weight=args.root_weight,
                    hierarchy_weight=args.hierarchy_weight,
                    cross_modal_weight=args.cross_modal_weight,
                    root_contrastive_weight=args.root_contrastive_weight,
                    level_contrastive_weight=args.level_contrastive_weight,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().item()
            if accelerator.sync_gradients:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}", depth=active_depth
                )

        metrics = evaluate(
            encoder,
            predictor,
            valid_loader,
            tokenizer,
            accelerator,
            args,
            level_num_classes,
        )
        metrics["epoch"] = epoch + 1
        metrics["active_depth"] = active_depth
        metrics["train_loss"] = running_loss / max(len(train_loader), 1)
        LOGGER.info("epoch=%d metrics=%s", epoch + 1, metrics)

        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            save_checkpoint(
                accelerator, encoder, predictor, args, metrics, "best_model.pt"
            )

    save_checkpoint(
        accelerator, encoder, predictor, args, metrics, "final_model.pt"
    )
    progress.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()
