import argparse
import logging
import math
import json
import os
import importlib.util

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from data_loader import VulDataLoader
from models.control_pathway import ControlPathway
from models.semantic_pathway import SemanticPathway
from models.feature_fusion_module import FeatureFusionModule
from models.transformer_llm_module import TransformerLLMModule
from models.multi_task_predictor import MultiTaskPredictor

logger = get_logger(__name__, log_level="INFO")

def parse_all_args():
    parser = argparse.ArgumentParser(description="Vulnerability Detection.")
    
    # 路径与数据相关
    parser.add_argument("--train_pt_path", type=str, required=True, help="Path to training pt.")
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to training csv.")
    parser.add_argument("--valid_pt_path", type=str, required=True, help="Path to valid pt.")
    parser.add_argument("--valid_csv_path", type=str, required=True, help="Path to valid csv.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to graph node vocabulary.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory for model predictions and checkpoints.")
    parser.add_argument("--run_name", type=str, default="vul_detect", help="Run name for logging and output folder.")
    
    # 模型配置
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3.5-4B", help="Base LLM for Semantic Pathway.")
    parser.add_argument("--transformer_base_name", type=str, default="microsoft/graphcodebert-base", help="Base Transformer for deep sequence modeling.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length for LLM inputs.")
    parser.add_argument("--max_safe_lines", type=int, default=200, help="Maximum number of code lines to consider for training.")
    parser.add_argument("--f_out_dim", type=int, default=128, help="Output dimension for both pathways before fusion.")
    parser.add_argument("--graph_embed_dim", type=int, default=128, help="GAT node embedding dimension.")
    parser.add_argument("--graph_hidden_dim", type=int, default=256, help="GAT hidden layer dimension.")
    parser.add_argument("--k_dim", type=int, default=256, help="Dimension of final deep representation k.")
    
    # 训练超参数
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. Overrides epochs if set.")
    parser.add_argument("--validation_steps", type=int, default=3000, help="Run validation every X steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate before backward pass.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts", help="Scheduler type (linear, cosine, constant, etc).")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Multi-task loss (Function level).")
    parser.add_argument("--beta", type=float, default=0.001, help="Weight for Contrastive loss in Feature Fusion.")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for Contrastive learning.")
    
    # 优化器与加速器
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mixed precision mode.")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on Ampere GPUs.")
    
    # 日志与断点
    parser.add_argument("--logging_dir", type=str, default="./logs", help="TensorBoard/Wandb log directory.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Integration to report logs to ('tensorboard', 'wandb').")
    parser.add_argument("--checkpointing_steps", type=int, default=10000, help="Save a checkpoint every X updates.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from, or 'latest'.")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    return args
def calc_metrics(tp, fp, tn, fn):
    eps = 1e-9
    pre = tp / (tp + fp + eps)
    re = tp / (tp + fn + eps)
    f1 = 2 * pre * re / (pre + re + eps)
    mcc_num = tp * tn - fp * fn
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_num / (mcc_den + eps)
    return mcc, pre, re, f1

def run_evaluation(models_list, valid_dataloader, tokenizer, accelerator, device, args, global_step, logger):
    control_model, semantic_model, fusion_model, transformer_model, predictor_model = models_list

    for model_module in models_list:
        model_module.eval()

    val_loss = 0.0
    total_func_samples = 0
    total_vul_funcs = 0

    # function-level confusion matrix
    f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0

    # statement-level confusion matrix
    one_s_tp, one_s_fp, one_s_tn, one_s_fn = 0, 0, 0, 0
    two_s_tp, two_s_fp, two_s_tn, two_s_fn = 0, 0, 0, 0

    # ranking metrics
    top1_hits, top3_hits, top5_hits = 0, 0, 0
    sum_mfr, sum_mar = 0.0, 0.0

    def calc_metrics(tp, fp, tn, fn):
        eps = 1e-9
        pre = tp / (tp + fp + eps)
        re = tp / (tp + fn + eps)
        f1 = 2 * pre * re / (pre + re + eps)

        mcc_num = tp * tn - fp * fn
        mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_num / (mcc_den + eps)
        return mcc, pre, re, f1

    with torch.no_grad():
        for _, val_batch in enumerate(valid_dataloader):
            batch_size = val_batch.num_graphs
            total_func_samples += batch_size

            val_line_mask_3d = val_batch.line_mask.view(
                batch_size, args.max_safe_lines, args.max_seq_len
            )
            val_y_s_2d = val_batch.y_s_dense.view(
                batch_size, args.max_safe_lines
            )

            # 真实行mask来自code_text的物理行数 而不是 line_mask
            real_line_mask = torch.zeros(
                (batch_size, args.max_safe_lines),
                dtype=torch.bool,
                device=device
            )
            for i, code_text in enumerate(val_batch.code_text):
                n_real_lines = min(len(str(code_text).split("\n")), args.max_safe_lines)
                real_line_mask[i, :n_real_lines] = True

            val_tok_code = tokenizer(
                val_batch.code_text,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt"
            ).to(device)

            val_tok_exp = tokenizer(
                val_batch.exp_text,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="pt"
            ).to(device)

            val_F_G_seq, graph_mask = control_model(
                x=val_batch.x,
                ast_edge_index=val_batch.ast_edge_index,
                cfg_edge_index=val_batch.cfg_edge_index,
                batch=val_batch.batch
            )
            val_F_C, val_F_C_global = semantic_model(
                code_ids=val_tok_code["input_ids"],
                exp_ids=val_tok_exp["input_ids"],
                code_mask=val_tok_code["attention_mask"],
                exp_mask=val_tok_exp["attention_mask"],
                is_training=True
            )

            val_h_m, val_L_ag = fusion_model(
                F_G_seq=val_F_G_seq,
                F_C=val_F_C,
                graph_mask=graph_mask,
                F_C_global=val_F_C_global,
                is_training=True
            )
            val_final_k = transformer_model(
                h_m=val_h_m,
                attention_mask=val_tok_code["attention_mask"]
            )

            # 模型前向仍然使用 line_mask
            val_prob_f, val_prob_s, v_loss = predictor_model(
                final_k=val_final_k,
                line_mask=val_line_mask_3d,
                is_training=True,
                y_f=val_batch.y_f,
                y_s=val_y_s_2d,
                L_ag=val_L_ag
            )

            avg_v_loss = accelerator.gather(v_loss.repeat(batch_size)).mean()
            val_loss += avg_v_loss.item()

            preds_f = (val_prob_f > 0.5).float()
            labels_f = val_batch.y_f.float()

            # function-level
            f_tp += ((preds_f == 1) & (labels_f == 1)).sum().item()
            f_fp += ((preds_f == 1) & (labels_f == 0)).sum().item()
            f_tn += ((preds_f == 0) & (labels_f == 0)).sum().item()
            f_fn += ((preds_f == 0) & (labels_f == 1)).sum().item()

            for i in range(batch_size):
                mask_i = real_line_mask[i]

                y_true = val_y_s_2d[i][mask_i].float()
                line_probs = val_prob_s[i][mask_i]
                raw_pred = (line_probs > 0.5).float()

                # one-phase: 只在真实漏洞函数上统计
                if labels_f[i] == 1:
                    one_s_tp += ((raw_pred == 1) & (y_true == 1)).sum().item()
                    one_s_fp += ((raw_pred == 1) & (y_true == 0)).sum().item()
                    one_s_tn += ((raw_pred == 0) & (y_true == 0)).sum().item()
                    one_s_fn += ((raw_pred == 0) & (y_true == 1)).sum().item()

                # two-phase
                if preds_f[i] == 0:
                    two_pred = torch.zeros_like(raw_pred)
                else:
                    two_pred = raw_pred

                two_s_tp += ((two_pred == 1) & (y_true == 1)).sum().item()
                two_s_fp += ((two_pred == 1) & (y_true == 0)).sum().item()
                two_s_tn += ((two_pred == 0) & (y_true == 0)).sum().item()
                two_s_fn += ((two_pred == 0) & (y_true == 1)).sum().item()

                # ranking仅在真实存在漏洞行的函数上统计
                true_lines_local = torch.nonzero(y_true).squeeze(-1)
                if len(true_lines_local) > 0:
                    total_vul_funcs += 1

                    sorted_indices = torch.argsort(line_probs, descending=True).tolist()
                    ranks = [sorted_indices.index(tl) + 1 for tl in true_lines_local.tolist()]

                    first_rank = min(ranks)
                    avg_rank = sum(ranks) / len(ranks)

                    sum_mfr += first_rank
                    sum_mar += avg_rank

                    if first_rank <= 1:
                        top1_hits += 1
                    if first_rank <= 3:
                        top3_hits += 1
                    if first_rank <= 5:
                        top5_hits += 1

    avg_val_loss = val_loss / max(len(valid_dataloader), 1)
    val_func_acc = (f_tp + f_tn) / max(total_func_samples, 1)

    f_mcc, f_pre, f_re, f_f1 = calc_metrics(f_tp, f_fp, f_tn, f_fn)
    one_mcc, one_pre, one_re, one_f1 = calc_metrics(one_s_tp, one_s_fp, one_s_tn, one_s_fn)
    two_mcc, two_pre, two_re, two_f1 = calc_metrics(two_s_tp, two_s_fp, two_s_tn, two_s_fn)

    top1_acc = top1_hits / total_vul_funcs if total_vul_funcs > 0 else 0.0
    top3_acc = top3_hits / total_vul_funcs if total_vul_funcs > 0 else 0.0
    top5_acc = top5_hits / total_vul_funcs if total_vul_funcs > 0 else 0.0
    mfr = sum_mfr / total_vul_funcs if total_vul_funcs > 0 else 0.0
    mar = sum_mar / total_vul_funcs if total_vul_funcs > 0 else 0.0

    log_metrics = {
        "val_loss": avg_val_loss,
        "val_func_acc": val_func_acc,
        "func_mcc": f_mcc,
        "func_pre": f_pre,
        "func_re": f_re,
        "func_f1": f_f1,
        "one_stmt_mcc": one_mcc,
        "one_stmt_pre": one_pre,
        "one_stmt_re": one_re,
        "one_stmt_f1": one_f1,
        "two_stmt_mcc": two_mcc,
        "two_stmt_pre": two_pre,
        "two_stmt_re": two_re,
        "two_stmt_f1": two_f1,
        "val_top1_acc": top1_acc,
        "val_top3_acc": top3_acc,
        "val_top5_acc": top5_acc,
        "val_mfr": mfr,
        "val_mar": mar,
    }

    final_score = (two_f1 * 0.1) + (top1_acc * 0.4) + (top3_acc * 0.3) + (top5_acc * 0.2)
    log_metrics["val_composite_score"] = final_score

    accelerator.log(log_metrics, step=global_step)

    logger.info(
        f"[Val {global_step}] "
        f"loss={avg_val_loss:.4f} "
        f"func_acc={val_func_acc:.4f}\n "
        f"func_mcc={f_mcc:.4f} func_pre={f_pre:.4f} func_re={f_re:.4f} func_f1={f_f1:.4f}\n "
        f"one_stmt_mcc={one_mcc:.4f} one_stmt_pre={one_pre:.4f} one_stmt_re={one_re:.4f} one_stmt_f1={one_f1:.4f}\n "
        f"two_stmt_mcc={two_mcc:.4f} two_stmt_pre={two_pre:.4f} two_stmt_re={two_re:.4f} two_stmt_f1={two_f1:.4f}\n "
        f"top1={top1_acc:.4f} top3={top3_acc:.4f} top5={top5_acc:.4f}\n "
        f"mfr={mfr:.4f} mar={mar:.4f}\n "
        f"composite={final_score:.4f}"
    )

    torch.cuda.empty_cache()
    for model_module in models_list:
        model_module.train()

    return final_score


def main():
    args = parse_all_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # 初始化Accelerator
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    device = accelerator.device

    # 日志与随机种子配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 数据加载
    logger.info("Loading training datasets...")

    data_loader_manager = VulDataLoader(
        pt_path=args.train_pt_path, 
        csv_path=args.train_csv_path, 
        batch_size=args.batch_size,
        max_safe_lines=args.max_safe_lines,
        num_workers=args.num_workers
    )
    train_dataloader = data_loader_manager.get_dataloader(shuffle=True, drop_last=True)
    
    logger.info("Loading validation dataset...")
    valid_loader_manager = VulDataLoader(
        pt_path=args.valid_pt_path, 
        csv_path=args.valid_csv_path, 
        batch_size=args.batch_size,
        max_safe_lines=args.max_safe_lines,
        num_workers=args.num_workers
    )
    valid_dataloader = valid_loader_manager.get_dataloader(shuffle=False, drop_last=False)

    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        graph_vocab_size = len(json.load(f))

    # 核心模型
    logger.info("Building Dual-Pathway Architecture...")
    
    control_model = ControlPathway(
        vocab_size=graph_vocab_size,
        embedding_dim=args.graph_embed_dim,
        hidden_features=args.graph_hidden_dim,
        out_features=args.f_out_dim
    )

    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map=None
    )
    real_llm_embedding = qwen_model.get_input_embeddings().to(torch.float32)
    semantic_model = SemanticPathway(
        embedding_layer=real_llm_embedding,
        hidden_size=qwen_model.config.hidden_size,  
        out_features=args.f_out_dim
    )

    semantic_model.shared_embedding.weight.requires_grad = False
    semantic_model.shared_embedding.eval()

    del qwen_model
    torch.cuda.empty_cache()

    fusion_model = FeatureFusionModule(hidden_dim=args.f_out_dim, tau=args.tau)
    transformer_model = TransformerLLMModule(hidden_dim=args.f_out_dim, k_dim=args.k_dim, model_name=args.transformer_base_name)
    predictor_model = MultiTaskPredictor(k_dim=args.k_dim, alpha=args.alpha, beta=args.beta)

    models_list = [control_model, semantic_model, fusion_model, transformer_model, predictor_model]

    # 断点保存与钩子函数注册
    def save_model_hook(models, weights, output_dir):
        # Accelerate保存钩子函数
        if accelerator.is_main_process:
            state_dict = {
                'control': models[0].state_dict(),
                'semantic': models[1].state_dict(),
                'fusion': models[2].state_dict(),
                'transformer': models[3].state_dict(),
                'predictor': models[4].state_dict()
            }
            torch.save(state_dict, os.path.join(output_dir, "vul_models.pt"))
            while len(weights) > 0:
                weights.pop()

    def load_model_hook(models, input_dir):
        # Accelerate加载钩子函数
        load_path = os.path.join(input_dir, "vul_models.pt")
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            models[0].load_state_dict(state_dict['control'])
            models[1].load_state_dict(state_dict['semantic'])
            models[2].load_state_dict(state_dict['fusion'])
            models[3].load_state_dict(state_dict['transformer'])
            models[4].load_state_dict(state_dict['predictor'])
            while len(models) > 0:
                models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # 硬件优化设置
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 优化器构建
    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        if importlib.util.find_spec("bitsandbytes") is not None:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Successfully loaded 8-bit Adam from bitsandbytes.")
        else:
            logger.warning("bitsandbytes is not installed. Falling back to standard AdamW.")

    all_parameters = (
        list(control_model.parameters()) + list(semantic_model.parameters()) +
        list(fusion_model.parameters()) + list(transformer_model.parameters()) +
        list(predictor_model.parameters())
    )
    
    optimizer = optimizer_cls(
        all_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Accelerator Prepare
    logger.info("Preparing components with Accelerator...")
    control_model, semantic_model, fusion_model, transformer_model, predictor_model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
        control_model, semantic_model, fusion_model, transformer_model, predictor_model, optimizer, train_dataloader, valid_dataloader
    )

    # 用分片后的dataloader长度计算步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.run_name, config=tracker_config)

    # Checkpoint 恢复
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # 训练主循环
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training Steps")

    best_composite_score = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        for model_module in models_list:
            model_module.train()
            
        train_loss = 0.0

        for step, batch_data in enumerate(train_dataloader):
            # 恢复训练步数跳过逻辑
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # Accelerator梯度累积上下文管理器
            with accelerator.accumulate(control_model, semantic_model, fusion_model, transformer_model, predictor_model):
                
                # 文本Tokenize并放置到设备
                tokenized_code = tokenizer(
                    batch_data.code_text, padding='max_length', truncation=True, 
                    max_length=args.max_seq_len, return_tensors='pt'
                ).to(device)
                
                tokenized_exp = tokenizer(
                    batch_data.exp_text, padding='max_length', truncation=True, 
                    max_length=args.max_seq_len, return_tensors='pt'
                ).to(device)

                # Control Pathway
                F_G_seq, graph_mask = control_model(
                    x=batch_data.x, 
                    ast_edge_index=batch_data.ast_edge_index, 
                    cfg_edge_index=batch_data.cfg_edge_index, 
                    batch=batch_data.batch
                )
                # Semantic Pathway
                F_C, F_C_global = semantic_model(
                    code_ids=tokenized_code['input_ids'], exp_ids=tokenized_exp['input_ids'],
                    code_mask=tokenized_code['attention_mask'], exp_mask=tokenized_exp['attention_mask'],
                    is_training=True
                )
                # Fusion Module
                h_m, L_ag = fusion_model(F_G_seq=F_G_seq, F_C=F_C, graph_mask=graph_mask, F_C_global=F_C_global, is_training=True)
                
                # Transformer
                final_k = transformer_model(h_m=h_m, attention_mask=tokenized_code['attention_mask'])
                
                # 将 PyG 拼接的二维张量重塑回三维
                batch_size = batch_data.num_graphs
                
                y_s_2d = batch_data.y_s_dense.view(batch_size, args.max_safe_lines)

                # [Batch * 行数, Token] -> [Batch, 行数, Token]
                line_mask_3d = batch_data.line_mask.view(batch_size, args.max_safe_lines, args.max_seq_len)
                
                # Predictor
                _, _, loss = predictor_model(
                    final_k=final_k,
                    line_mask=line_mask_3d,
                    is_training=True,
                    y_f=batch_data.y_f,
                    y_s=y_s_2d,
                    L_ag=L_ag
                )

                # 记录损失并在分布式环境中收集
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # 反向传播与优化
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_parameters, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 实际发生一次梯度更新时
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # TensorBoard / Wandb 日志记录
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                train_loss = 0.0

                if global_step % args.validation_steps == 0 or global_step == 10:
                    logger.info(f"Running validation at step {global_step}...")

                    current_score = run_evaluation(
                        models_list=models_list, 
                        valid_dataloader=valid_dataloader, 
                        tokenizer=tokenizer, 
                        accelerator=accelerator, 
                        device=device, 
                        args=args, 
                        global_step=global_step,
                        logger=logger
                    )

                    if current_score > best_composite_score:
                        best_composite_score = current_score
                        if accelerator.is_main_process:
                            best_save_path = os.path.join(args.output_dir, "best_model")
                            accelerator.save_state(best_save_path)
                            logger.info(f"Saved new best model with Composite Score: {best_composite_score:.4f} to {best_save_path}")

                # Checkpoint 保存逻辑
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    

            # tqdm
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
                
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()

    print("Training complete!")

    result_path = os.path.join(args.output_dir, "result.json")

    with open(result_path, "w") as f:
        json.dump({
            "score": best_composite_score
        }, f)


if __name__ == "__main__":
    main()