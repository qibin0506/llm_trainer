from typing import Tuple, List, Optional
from contextlib import nullcontext
import gc
import math
from itertools import islice

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .dataset import RLDataset
from .loss import GRPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .log import Logger
from .train_configs import TrainConfig, RewardFun, GenerationService
from .utils import (
    autocast,
    left_pad_sequence,
    log_softmax,
    disable_dropout_in_model,
    calc_position_ids,
    empty_cache,
    fold_rewards_to_preceding_action,
    build_completion_mask
)
from .checkpoint import save_checkpoint, save_steps, restore_rng_state_dict
from .partition_utils import unwrap_model_for_generation, zero3_trace_guard, add_fused_log_prob_head


class GRPOTrainer(BaseTrainer):
    """
    GRPOTrainer

    Args:
        train_config:
            - 全局训练配置，必须包含 grpo_config。

        reward_func:
            - 基于 Rule 规则或 RM 模型的奖励打分函数，支持 1D 标量或 2D 逐 Token 序列奖励。

        generation_service:
            - 外部自定义生成服务接口

        eval_prompts:
            - 评估测试的提示词列表。
            - [num_eval_prompts] 长度的字符串列表。
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_func: RewardFun,
            generation_service: Optional[GenerationService] = None,
            eval_prompts: List[str]
    ):
        self.grpo_config = train_config.grpo_config
        self.chunked_log_probs_size = self.grpo_config.chunked_log_probs_size
        self.rollout_eval_batch_size = self.grpo_config.rollout_eval_batch_size
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            generation_service=generation_service,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            return_logits=self.chunked_log_probs_size is None
        )

        grpo_batch_size = self.grpo_config.grpo_batch_size
        total_generated_seqs = train_config.batch_size * self.grpo_config.group_size
        assert total_generated_seqs % (grpo_batch_size * self.gradient_accumulation_steps) == 0, \
            '(batch_size * group_size) % (grpo_batch_size * gradient_accumulation_steps) must be zero!'

        self.reward_func = reward_func
        self.ref_model = self._init_ref_model()
        self.criterion = self._init_loss()

        if self.grpo_config.loss_type == "luspo" and self.grpo_config.loss_importance_sampling_level != "sequence":
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    "WARN: When using 'luspo' loss, `loss_importance_sampling_level` should ideally "
                    "be set to 'sequence' to properly mirror the Length-Bias mitigation setup from the LUSPO paper."
                )

        # 校验 VESPO 的配置冗余
        if self.grpo_config.loss_type == "vespo" and self.grpo_config.loss_importance_sampling_level != "token":
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    "WARN: VESPO computes sequence-level importance weights internally. "
                    "Your `loss_importance_sampling_level` setting will be ignored for the Gamma weights computation."
                )

        # 校验 CISPO 的 loss_clip_eps_high 配置合理性（CISPO 中该参数代表比率绝对上界，推荐值为 5.0）
        if self.grpo_config.loss_type == "cispo":
            eff_eps_high = (
                self.grpo_config.loss_clip_eps_high
                if self.grpo_config.loss_clip_eps_high is not None
                else self.grpo_config.loss_clip_eps
            )
            if eff_eps_high <= 1.0 and TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: `loss_type` is 'cispo', where `loss_clip_eps_high` represents the absolute upper bound "
                    f"for importance sampling ratios (recommended: 5.0 in ScaleRL), but it is currently set to "
                    f"{eff_eps_high} (<= 1.0). This will clip all normal ratios (~1.0). Please verify your config."
                )

        # 校验 clip_eps 与重要性采样级别的匹配性
        if self.grpo_config.loss_importance_sampling_level == "sequence" and self.grpo_config.loss_clip_eps > 0.05:
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: `loss_importance_sampling_level` is 'sequence' (e.g. GSPO/LUSPO), but `loss_clip_eps` is set to "
                    f"{self.grpo_config.loss_clip_eps} (> 0.05). For sequence-level importance sampling, smaller epsilons "
                    f"(such as 3e-4 ~ 4e-4) are standard to prevent ratio explosion. Please verify your config."
                )
        elif self.grpo_config.loss_importance_sampling_level == "token" and self.grpo_config.loss_clip_eps < 1e-2:
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: `loss_importance_sampling_level` is 'token', but `loss_clip_eps` is set to "
                    f"{self.grpo_config.loss_clip_eps} (< 0.01). This will cause severe over-clipping for token-level GRPO. "
                    f"Consider using standard clip epsilon (e.g. 0.2)."
                )

        # 校验生成采样参数与策略求值分布的一致性
        gen_cfg = self.grpo_config.generate_config
        if gen_cfg.top_p < 1.0 or (gen_cfg.top_k is not None and gen_cfg.top_k > 0) or gen_cfg.repetition_penalty != 1.0:
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: Rollout generate_config uses truncation or penalties (top_p={gen_cfg.top_p}, "
                    f"top_k={gen_cfg.top_k}, repetition_penalty={gen_cfg.repetition_penalty}). "
                    f"This creates discrepancy between the sampling distribution and the un-truncated log_softmax evaluation. "
                    f"For on-policy RL, top_p=1.0, top_k=None, and repetition_penalty=1.0 are recommended."
                )

        if (self.grpo_config.loss_importance_sampling_level == "sequence"
                and self.grpo_config.loss_type in ["bnpo", "dr_grpo", "cispo", "dapo"]):
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: `loss_importance_sampling_level='sequence'` (e.g. GSPO) is theoretically conflicting with "
                    f"token-level normalization in '{self.grpo_config.loss_type}'. For sequence-level importance sampling, "
                    f"use standard 'grpo' or 'luspo'."
                )

    def _init_ref_model(self):
        # beta == 0，不需要ref_model
        if self.grpo_config.loss_beta == 0.0:
            return None

        if not self.grpo_config.ref_model_weights_path:
            raise ValueError(
                "GRPO `loss_beta` > 0 requires a reference model to compute KL divergence, "
                "but `ref_model_weights_path` is not configured! "
                "Please configure `ref_model_weights_path` in GRPOConfig to prevent using randomly initialized weights in distributed training."
            )

        parallel_kwargs = self._init_ref_model_args(self.train_config.model_config)
        with self._new_model_context(parallel_kwargs):
            ref_model = self._new_model(self.train_config)

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        self._load_external_weights(ref_model, self.grpo_config.ref_model_weights_path)
        self._check_add_fused_head(ref_model)

        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,
            kwargs=parallel_kwargs,
            save_instance=False
        )

        return ref_model

    def _new_model(self, train_config: TrainConfig):
        model = super()._new_model(train_config)
        disable_dropout_in_model(model)
        return model

    def _check_add_fused_head(self, model):
        if self.chunked_log_probs_size is not None:
            add_fused_log_prob_head(model, chunk_size=self.chunked_log_probs_size)

    def _init_loss(self):
        grpo_criterion = GRPOLoss(
            beta=self.grpo_config.loss_beta,
            clip_eps_low=self.grpo_config.loss_clip_eps,
            clip_eps_high=self.grpo_config.loss_clip_eps_high,
            delta=self.grpo_config.loss_delta,
            importance_sampling_level=self.grpo_config.loss_importance_sampling_level,
            loss_type=self.grpo_config.loss_type,
            sapo_temperature_pos=self.grpo_config.sapo_temperature_pos,
            sapo_temperature_neg=self.grpo_config.sapo_temperature_neg,
            vespo_k_pos=self.grpo_config.vespo_k_pos,
            vespo_lambda_pos=self.grpo_config.vespo_lambda_pos,
            vespo_k_neg=self.grpo_config.vespo_k_neg,
            vespo_lambda_neg=self.grpo_config.vespo_lambda_neg,
            use_bias_correction_kl=self.grpo_config.use_bias_correction_kl,
        )

        return grpo_criterion

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()

        if parallel_kwargs:
            parallel_kwargs['train_micro_batch_size_per_gpu'] = self.grpo_config.grpo_batch_size

        data_loader_kwargs.update({"collate_fn": lambda x: x})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        return RLDataset(file_path), file_path

    def _compute_completion_log_probs(
            self,
            model,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            completion_ids: torch.Tensor,
            completion_mask: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算 Completion 部分的 LogProbs：
        1. 采用 Active Token Filtering：跳过 Prompt、Padding 及多轮非 Action (Feedback) 的 lm_head 矩阵乘法
        2. 支持 batch_size 样本维度切片：每个子 batch 独立包裹 zero3_trace_guard，杜绝多子批 no_grad 前向下的 Trace 累积
        """
        is_no_grad = not torch.is_grad_enabled()
        use_fused = (self.chunked_log_probs_size is not None)

        def _step_forward(inp_ids, att_mask, comp_ids, comp_mask):
            guard = zero3_trace_guard(model) if is_no_grad else nullcontext()
            with guard:
                pos_ids = calc_position_ids(att_mask)
                comp_len = comp_ids.size(1)

                temperature = self.grpo_config.generate_config.temperature or 1.0
                outputs = model(
                    input_ids=inp_ids,
                    attention_mask=att_mask,
                    position_ids=pos_ids,
                    completion_ids=comp_ids if use_fused else None,
                    completion_mask=comp_mask if use_fused else None,
                    temperature=temperature,
                    fused_log_probs=use_fused,
                    return_logits=not use_fused
                )

                if isinstance(outputs, dict) and 'log_probs' in outputs:
                    log_probs = outputs['log_probs']
                    if log_probs.size(1) > comp_len:
                        log_probs = log_probs[:, -comp_len:]
                else:
                    logits_completion = outputs['logits'][:, :-1, :][:, -comp_len:, :].float()
                    if temperature != 1.0:
                        logits_completion = logits_completion / temperature
                    log_probs = log_softmax(logits_completion, comp_ids)
                aux_loss = outputs.get('aux_loss', None)

                return log_probs, aux_loss

        if batch_size is None or batch_size <= 0 or batch_size >= input_ids.size(0):
            return _step_forward(input_ids, attention_mask, completion_ids, completion_mask)

        all_log_probs = []
        aux_losses = []
        for i in range(0, input_ids.size(0), batch_size):
            c_input_ids = input_ids[i: i + batch_size]
            c_attention_mask = attention_mask[i: i + batch_size] if attention_mask is not None else None
            c_completion_ids = completion_ids[i: i + batch_size]
            c_completion_mask = completion_mask[i: i + batch_size] if completion_mask is not None else None
            log_probs, aux_loss = _step_forward(c_input_ids, c_attention_mask, c_completion_ids, c_completion_mask)
            all_log_probs.append(log_probs)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        total_log_probs = torch.cat(all_log_probs, dim=0)
        total_aux_loss = sum(aux_losses) / len(aux_losses) if len(aux_losses) > 0 else None

        return total_log_probs, total_aux_loss

    def _compute_group_relative_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        group_size = self.grpo_config.group_size
        rewards_fp32 = rewards.float()

        # Reshape rewards to group by prompt: [batch, group_size]
        rewards_by_group = rewards_fp32.view(-1, group_size)

        # 使用 nanmean，仅基于组内有效样本计算均值，排除 NaN 干扰
        # 当某组全为 NaN 时，nanmean 得到 NaN，nan_to_num 兜底为 0.0
        group_means = torch.nan_to_num(torch.nanmean(rewards_by_group, dim=1, keepdim=True), nan=0.0)

        scale_rewards = self.grpo_config.scale_rewards
        if scale_rewards is None:
            scale_rewards = (self.grpo_config.loss_type != "dr_grpo")

        if scale_rewards:
            # 使用无偏 (unbiased=True) 估计计算组内有效标准差
            valid_mask = ~torch.isnan(rewards_by_group)
            valid_counts = valid_mask.sum(dim=1, keepdim=True)

            diff = torch.where(valid_mask, rewards_by_group - group_means, torch.zeros_like(rewards_by_group))
            variance = (diff ** 2).sum(dim=1, keepdim=True) / (valid_counts - 1).clamp(min=1.0)
            group_stds = torch.sqrt(variance)

            # 当组内有效样本 < 2 或标准差极小时，不进行 std 缩放
            is_valid = (valid_counts >= 2) & (group_stds > 1e-6)
            norm_adv = (rewards_by_group - group_means) / (group_stds + 1e-8)
            advantages = torch.where(is_valid, norm_adv, torch.zeros_like(rewards_by_group))
        else:
            advantages = rewards_by_group - group_means

        # 最终将包含 NaN 的无效样本优势置为 0.0，杜绝产生错误梯度
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        return advantages.view_as(rewards).to(dtype=rewards.dtype).unsqueeze(1)

    def _compute_multi_turn_advantages(
            self,
            rewards_tensor: torch.Tensor,
            loss_mask: torch.Tensor,
            gamma: float = 1.0
    ) -> torch.Tensor:
        """
        针对多轮交互 Agent 的轨迹级整体评估。

        - 轨迹级评估: 每个候选样本在整条多轮轨迹结束时给出一个全局评分 R_i (若为 2D 稠密奖励则累加属于 Action 的有效增量)；
        - 组内相对标准化: 基于同 Prompt 组计算均值与标准差，消除尺度与幸存者偏差；
        - 共享优势: 整条轨迹的所有有效 Action Token 共享该标准化优势，彻底解决多轮 Return-to-Go 量纲与时间视界失配问题。
        """
        if rewards_tensor.dim() == 2:
            masked_rewards = torch.where(
                loss_mask,
                rewards_tensor.float(),
                torch.zeros_like(rewards_tensor, dtype=torch.float32)
            )
            trajectory_rewards = masked_rewards.sum(dim=-1)
        elif rewards_tensor.dim() == 1:
            trajectory_rewards = rewards_tensor.float()
        else:
            raise ValueError(f"Unsupported rewards_tensor dim: {rewards_tensor.dim()}, expected 1 or 2.")

        # 标准组内相对优势计算，返回 [batch_size, 1] 标量优势
        return self._compute_group_relative_advantages(trajectory_rewards)


    def _compute_token_level_prm_advantages(
            self,
            rewards_tensor: torch.Tensor,
            loss_mask: torch.Tensor,
            gamma: float = 1.0
    ) -> torch.Tensor:
        """
        针对单轮/多轮推理任务中 2D Token-level / Step-level PRM 过程奖励的细粒度因果时序信用分配。

        1. 沿时序从后向前反向累积计算各 Token 的因果未来折现回报 (Reward-to-Go):
           G_{i, t} = r_{i, t} + gamma * G_{i, t+1}
        2. 按同 Prompt 组在各时步活跃 Token 之间进行组内归一化：
           - 处于活跃状态的候选根据其 Reward-to-Go 进行组内标准化；
           - 已提前终止的候选在后续时步因果未来回报自然为 0.0，避免末步单 Token 奖励拉低基线导致长序列获得虚假正优势。
        3. 保留 Token-level 细粒度优势分布，彻底解决 PRM 奖励被粗暴求和退化为标量 ORM 的缺陷。
        """
        batch_size, seq_len = rewards_tensor.shape
        group_size = self.grpo_config.group_size
        device = rewards_tensor.device

        # 使用 torch.where 屏蔽非动作位置，避免 NaN * 0.0 污染
        masked_rewards = torch.where(
            loss_mask,
            rewards_tensor.float(),
            torch.zeros_like(rewards_tensor, dtype=torch.float32)
        )

        # 1. 反向累积因果 Reward-to-Go：非动作 Token 仅透传未来回报
        r2g = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        running_r2g = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for t in range(seq_len - 1, -1, -1):
            is_act = loss_mask[:, t]
            running_r2g = torch.where(is_act, masked_rewards[:, t] + gamma * running_r2g, running_r2g)
            r2g[:, t] = torch.where(is_act, running_r2g, torch.zeros_like(running_r2g))

        # 2. 组内归一化：针对因果未来时序回报计算组相对优势
        r2g_by_group = r2g.view(-1, group_size, seq_len)
        valid_mask_by_group = loss_mask.view(-1, group_size, seq_len) & (~torch.isnan(r2g_by_group))

        # 对于已提前终止的短序列，在其终止后的未来时间步，其因果未来折现回报自然为 0.0。
        # 严禁错误地使用末步单 Token 奖励 (r_L) 作为常数填充拉低基准，导致劣质长序列尾部获得虚假正优势。
        fallback_r2g = torch.where(valid_mask_by_group, r2g_by_group, torch.zeros_like(r2g_by_group))
        has_any_valid = valid_mask_by_group.any(dim=-1, keepdim=True)
        baseline_mask = (~torch.isnan(fallback_r2g)) & has_any_valid
        valid_counts = baseline_mask.float().sum(dim=1, keepdim=True)

        safe_r2g = torch.where(baseline_mask, fallback_r2g, torch.zeros_like(fallback_r2g))
        group_means = safe_r2g.sum(dim=1, keepdim=True) / valid_counts.clamp(min=1.0)
        diff_sq = torch.where(
            baseline_mask,
            (safe_r2g - group_means) ** 2,
            torch.zeros_like(safe_r2g)
        )
        group_vars = diff_sq.sum(dim=1, keepdim=True) / (valid_counts - 1.0).clamp(min=1.0)
        group_stds = torch.sqrt(group_vars)

        scale_rewards = self.grpo_config.scale_rewards
        if scale_rewards is None:
            scale_rewards = (self.grpo_config.loss_type != "dr_grpo")

        if scale_rewards:
            # 仅对当前有效生成 Token 计算非零优势
            is_valid = (valid_counts >= 2) & (group_stds > 1e-6) & valid_mask_by_group
            norm_adv = (r2g_by_group - group_means) / (group_stds + 1e-8)
            advantages = torch.where(is_valid, norm_adv, torch.zeros_like(r2g_by_group))
        else:
            advantages = torch.where(
                (valid_counts >= 2) & valid_mask_by_group,
                r2g_by_group - group_means,
                torch.zeros_like(r2g_by_group)
            )

        advantages = advantages.view(batch_size, seq_len)
        advantages = torch.where(loss_mask, advantages, torch.zeros_like(advantages))
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        return advantages.to(dtype=rewards_tensor.dtype)

    def _sample_rollout_completions(self, batch_data: List[dict]) -> dict:
        prompt_ids = [item["prompt"] for item in batch_data]
        gt_answer_ids = [item["answer"] for item in batch_data]
        group_size = self.grpo_config.group_size

        pad_token_id = TrainerTools().tokenizer.pad
        device = TrainerTools().parallel.device

        padded_prompt_ids = left_pad_sequence(prompt_ids, padding_value=pad_token_id)
        padded_prompt_ids = padded_prompt_ids.to(device)
        prompt_len = padded_prompt_ids.shape[1]

        # [batch*group_size, max_prompt_len]
        padded_prompt_ids = padded_prompt_ids.repeat_interleave(group_size, 0)
        prompt_masks = self._calc_attention_mask(padded_prompt_ids)

        max_seq_len = self.grpo_config.generate_config.max_seq_len
        if prompt_len >= max_seq_len:
            max_allowed_prompt_len = max(1, max_seq_len - 16)
            if TrainerTools().parallel.is_main_process:
                Logger.std_log(
                    f"WARN: [Rank {TrainerTools().parallel.global_rank}] Prompt length ({prompt_len}) >= max_seq_len ({max_seq_len}). "
                    f"Truncating prompt to last {max_allowed_prompt_len} tokens to leave generation budget and prevent rank abort."
                )
            padded_prompt_ids = padded_prompt_ids[:, -max_allowed_prompt_len:]
            prompt_masks = self._calc_attention_mask(padded_prompt_ids)
            prompt_len = padded_prompt_ids.shape[1]

        max_new_tokens = max(1, max_seq_len - prompt_len)

        external_gen_mask = None
        dones_list = None
        feedbacks_list = None
        padded_gen_masks = []
        with torch.no_grad():
            if self.generation_service is not None:
                service_output = self.generation_service(
                    self.train_model, padded_prompt_ids, self.grpo_config.generate_config,
                    'grpo', None, None
                )

                completion_ids_list = service_output['completions']
                dones_list = service_output.get('dones', None)
                feedbacks_list = service_output.get('feedbacks', None)
                gen_masks_list = service_output.get('generation_masks', None)

                if gen_masks_list is not None:
                    assert len(gen_masks_list) == len(completion_ids_list)

                padded_completions = []
                padded_gen_masks = []
                max_comp_len = max((len(c) for c in completion_ids_list), default=0)
                if TrainerTools().parallel.parallel_train and dist.is_initialized():
                    max_comp_len_t = torch.tensor(max_comp_len, dtype=torch.long, device=device)
                    dist.all_reduce(max_comp_len_t, op=dist.ReduceOp.MAX)
                    max_comp_len = int(max_comp_len_t.item())

                if max_comp_len == 0:
                    max_comp_len = 1

                if max_comp_len > max_new_tokens:
                    if TrainerTools().parallel.is_main_process:
                        Logger.std_log(
                            f"WARN: generation_service returned completions of length {max_comp_len} "
                            f"exceeding max_new_tokens ({max_new_tokens}). Completions will be truncated, "
                            f"and terminal status will be invalidated for truncated sequences."
                        )
                    if dones_list is not None:
                        for idx, comp in enumerate(completion_ids_list):
                            if len(comp) > max_new_tokens:
                                dones_list[idx] = False

                max_comp_len = min(max_comp_len, max_new_tokens)

                for idx, comp in enumerate(completion_ids_list):
                    comp = comp[:max_comp_len]
                    pad_len = max_comp_len - len(comp)
                    padded_completions.append(comp + [pad_token_id] * pad_len)

                    if gen_masks_list is not None and idx < len(gen_masks_list):
                        g_mask = gen_masks_list[idx][:max_comp_len]
                        padded_gen_masks.append(g_mask + [False] * pad_len)

                completion_ids = torch.tensor(padded_completions, dtype=torch.long, device=device)
                if padded_gen_masks:
                    external_gen_mask = torch.tensor(padded_gen_masks, dtype=torch.bool, device=device)

                TrainerTools().parallel.wait('rollout')
            else:
                with unwrap_model_for_generation(self.train_model) as unwrapped_model:
                    unwrapped_model.eval()
                    try:
                        gen_chunk_size = self.grpo_config.generate_config.chunked_generate_size
                        if gen_chunk_size is None or gen_chunk_size <= 0:
                            gen_chunk_size = padded_prompt_ids.size(0)

                        chunk_completions = []
                        max_comp_len = 0
                        for start_idx in range(0, padded_prompt_ids.size(0), gen_chunk_size):
                            chunk_prompts = padded_prompt_ids[start_idx: start_idx + gen_chunk_size]
                            chunk_masks = prompt_masks[start_idx: start_idx + gen_chunk_size]
                            chunk_out, _ = batch_generate(
                                model=unwrapped_model,
                                tokens=chunk_prompts,
                                attention_mask=chunk_masks,
                                max_new_tokens=max_new_tokens,
                                temperature=self.grpo_config.generate_config.temperature,
                                top_k=self.grpo_config.generate_config.top_k,
                                top_p=self.grpo_config.generate_config.top_p,
                                repetition_penalty=self.grpo_config.generate_config.repetition_penalty,
                                exclude_penalty_tokens=self.grpo_config.generate_config.exclude_penalty_tokens,
                                device=device,
                                suppress_tokens=self.grpo_config.generate_config.suppress_tokens,
                                return_logits=False,
                                auto_prefix_cache=self.grpo_config.generate_config.auto_prefix_cache
                            )
                            comp = chunk_out[:, prompt_len:]
                            max_comp_len = max(max_comp_len, comp.shape[1])
                            chunk_completions.append(comp)

                        # 跨卡全局同步最大补全长度：
                        # 1. 彻底解决各 Rank 生成结束时间不同、短序列 Rank 提前退出上下文引发的相移死锁 (Phase Mismatch)
                        # 2. 保证全卡 completion_ids 的 sequence_length 严格对齐，避免异构张量维度导致后续通信或计算错位
                        if TrainerTools().parallel.parallel_train and dist.is_initialized():
                            max_comp_len_t = torch.tensor(max_comp_len, dtype=torch.long, device=device)
                            dist.all_reduce(max_comp_len_t, op=dist.ReduceOp.MAX)
                            max_comp_len = int(max_comp_len_t.item())

                        if max_comp_len == 0:
                            if TrainerTools().parallel.is_main_process:
                                Logger.std_log(
                                    "WARN: All sequences generated 0 new tokens (empty completion). "
                                    "Padding completion length to 1 with pad_token_id to prevent negative slice crash."
                                )
                            max_comp_len = 1
                            chunk_completions = [
                                torch.full((comp.size(0), 1), pad_token_id, dtype=torch.long, device=device)
                                for comp in chunk_completions
                            ]

                        padded_comps = [
                            F.pad(comp, (0, max_comp_len - comp.shape[1]), value=pad_token_id)
                            if comp.shape[1] < max_comp_len else comp
                            for comp in chunk_completions
                        ]
                        completion_ids = torch.cat(padded_comps, dim=0)
                        dones_list = torch.any(completion_ids == TrainerTools().tokenizer.end, dim=1).cpu().tolist()

                        TrainerTools().parallel.wait('rollout')
                    finally:
                        unwrapped_model.train()

        return {
            'prompt_ids': prompt_ids,
            'gt_answer_ids': gt_answer_ids,
            'group_size': group_size,
            'padded_prompt_ids': padded_prompt_ids,
            'prompt_masks': prompt_masks,
            'completion_ids': completion_ids,
            'external_gen_mask': external_gen_mask,
            'dones_list': dones_list,
            'feedbacks_list': feedbacks_list,
            'padded_gen_masks': padded_gen_masks,
            'max_new_tokens': max_new_tokens,
        }

    def _evaluate_rollout_data(self, sample_data: dict) -> dict:
        prompt_ids = sample_data['prompt_ids']
        gt_answer_ids = sample_data['gt_answer_ids']
        group_size = sample_data['group_size']
        padded_prompt_ids = sample_data['padded_prompt_ids']
        prompt_masks = sample_data['prompt_masks']
        completion_ids = sample_data['completion_ids']
        external_gen_mask = sample_data['external_gen_mask']
        dones_list = sample_data['dones_list']
        feedbacks_list = sample_data['feedbacks_list']
        padded_gen_masks = sample_data['padded_gen_masks']
        max_new_tokens = sample_data['max_new_tokens']

        pad_token_id = TrainerTools().tokenizer.pad
        device = TrainerTools().parallel.device

        eos_token_id = TrainerTools().tokenizer.end
        if external_gen_mask is not None:
            # 多轮 Agent 模式沿用原 external_gen_mask 逻辑
            completion_pad_mask = (completion_ids != pad_token_id)
            loss_mask = completion_pad_mask & external_gen_mask
        else:
            # 单轮模式：使用精准的 EOS 截断与 pad==eos 保护
            loss_mask = build_completion_mask(completion_ids, eos_token_id, pad_token_id)
            completion_pad_mask = loss_mask

        input_ids = torch.cat([padded_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_masks, completion_pad_mask], dim=1)

        total_samples = input_ids.size(0)
        accum_steps = self.gradient_accumulation_steps
        samples_per_accum_window = self.grpo_config.grpo_batch_size * accum_steps
        can_skip_rollout_forward = (total_samples <= samples_per_accum_window)
        # 跨卡同步 can_skip_rollout_forward 布尔决策，防止多卡样本数不一致时 ZeRO-3 AllGather 死锁
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            skip_tensor = torch.tensor(1 if can_skip_rollout_forward else 0, dtype=torch.int32, device=device)
            dist.all_reduce(skip_tensor, op=dist.ReduceOp.MIN)
            can_skip_rollout_forward = bool(skip_tensor.item() == 1)

        eval_batch_size = self.rollout_eval_batch_size
        with torch.no_grad():
            if can_skip_rollout_forward:
                old_log_probs = None
            else:
                with autocast(TrainerTools().parallel.device_type):
                    old_log_probs, _ = self._compute_completion_log_probs(
                        self.train_model, input_ids, attention_mask, completion_ids,
                        completion_mask=loss_mask, batch_size=eval_batch_size
                    )

            if self.ref_model:
                with autocast(TrainerTools().parallel.device_type):
                    ref_log_probs, _ = self._compute_completion_log_probs(
                        self.ref_model, input_ids, attention_mask, completion_ids,
                        completion_mask=loss_mask, batch_size=eval_batch_size
                    )
            else:
                ref_log_probs = None

        repeated_prompt_ids = [p for p in prompt_ids for _ in range(group_size)]
        repeated_gt_answer_ids = [a for a in gt_answer_ids for _ in range(group_size)]

        extra_reward_kwargs = {}
        if dones_list is not None:
            extra_reward_kwargs['dones'] = dones_list
        if padded_gen_masks:
            extra_reward_kwargs['generation_masks'] = padded_gen_masks
        elif external_gen_mask is not None:
            extra_reward_kwargs['generation_masks'] = external_gen_mask.cpu()
        if feedbacks_list is not None:
            extra_reward_kwargs['feedbacks'] = feedbacks_list

        raw_rewards = self.reward_func(
            repeated_prompt_ids,
            completion_ids.cpu(),
            repeated_gt_answer_ids,
            **extra_reward_kwargs
        )
        if isinstance(raw_rewards, torch.Tensor):
            rewards_tensor = raw_rewards.to(dtype=torch.float32, device=TrainerTools().parallel.device)
        else:
            rewards_tensor = torch.tensor(
                raw_rewards,
                dtype=torch.float32,
                device=TrainerTools().parallel.device
            )

        # 计算每个样本在模型生成区域（action tokens）的最后一个有效 Token 位置与结束符判断
        eos_token_id = TrainerTools().tokenizer.end
        gen_indices = torch.where(
            loss_mask,
            torch.arange(completion_ids.size(1), device=device).unsqueeze(0),
            torch.tensor(-1, device=device)
        )
        last_token_indices = gen_indices.max(dim=1).values
        valid_indices_mask = last_token_indices >= 0
        batch_range = torch.arange(completion_ids.size(0), device=device)

        if external_gen_mask is not None:
            # 多轮 Agent 模式：以模型最后一个有效 action token 是否为 eos_token_id 判定是否正常结束
            last_action_tokens = torch.where(
                valid_indices_mask,
                completion_ids[batch_range, last_token_indices.clamp(min=0)],
                torch.tensor(pad_token_id, device=device)
            )
            has_eos = (last_action_tokens == eos_token_id) & valid_indices_mask
        else:
            # 单轮模式：只要 completion 中包含 eos_token_id 即视为正常结束
            has_eos = torch.any(completion_ids == eos_token_id, dim=1)

        if self.grpo_config.missing_eos_penalty is not None:
            penalty = self.grpo_config.missing_eos_penalty
            if rewards_tensor.dim() == 2:
                missing_eos_mask = (~has_eos) & valid_indices_mask
                if missing_eos_mask.any():
                    m_batch = batch_range[missing_eos_mask]
                    m_last = last_token_indices[missing_eos_mask]
                    rewards_tensor[m_batch, m_last] -= penalty
            elif rewards_tensor.dim() == 1:
                missing_eos_mask = (~has_eos) & valid_indices_mask
                if missing_eos_mask.any():
                    rewards_tensor[missing_eos_mask] -= penalty

        is_multi_turn = (external_gen_mask is not None)

        if is_multi_turn:
            if rewards_tensor.dim() == 1:
                # 多轮 RL 模式下接收 1D 轨迹标量奖励 R_i
                rewards = rewards_tensor
                advantages = self._compute_multi_turn_advantages(
                    rewards_tensor=rewards,
                    loss_mask=loss_mask,
                    gamma=self.grpo_config.gamma
                )
            else:
                assert rewards_tensor.shape == completion_ids.shape, (
                    f"2D dense reward shape {rewards_tensor.shape} must match completion_ids shape {completion_ids.shape}"
                )
                # 自动检查并折叠非 Action 位置（如 Feedback/Pad）的奖励至紧邻的前序 Action 最后一个 Token
                rewards_tensor = fold_rewards_to_preceding_action(
                    rewards_tensor,
                    loss_mask,
                    context_desc="GRPO"
                )
                masked_rewards = torch.where(loss_mask, rewards_tensor.float(), torch.zeros_like(rewards_tensor, dtype=torch.float32))
                rewards = masked_rewards.sum(dim=-1)

                if self.grpo_config.multi_turn_advantage_mode == 'sequence':
                    # 针对多轮交互 Agent 的轨迹级整体评估
                    # 每个候选样本在整条多轮轨迹结束时给出一个全局评分 R_i，同 Prompt 组内对齐计算标准化优势，
                    # 属于该样本的所有有效 Action Token 共享这一全局标量优势，彻底消除跨轮时序量纲失配与幸存者偏差
                    advantages = self._compute_multi_turn_advantages(
                        rewards_tensor=rewards,
                        loss_mask=loss_mask,
                        gamma=self.grpo_config.gamma
                    )
                else:
                    # 针对多轮交互中 2D Dense PRM 过程奖励的细粒度 Token/Step-level 时序因果信用分配
                    advantages = self._compute_token_level_prm_advantages(
                        rewards_tensor=rewards_tensor,
                        loss_mask=loss_mask,
                        gamma=self.grpo_config.gamma
                    )
        elif rewards_tensor.dim() == 2:
            assert rewards_tensor.shape == completion_ids.shape, (
                f"2D dense reward shape {rewards_tensor.shape} must match completion_ids shape {completion_ids.shape}"
            )
            dense_rewards = rewards_tensor
            masked_rewards = torch.where(loss_mask, dense_rewards.float(), torch.zeros_like(dense_rewards, dtype=torch.float32))
            rewards = masked_rewards.sum(dim=-1)

            advantages = self._compute_token_level_prm_advantages(
                rewards_tensor=dense_rewards,
                loss_mask=loss_mask,
                gamma=self.grpo_config.gamma
            )
        elif rewards_tensor.dim() == 1:
            # 单轮模式 1D 标量轨迹奖励：标准组内相对优势计算
            rewards = rewards_tensor
            advantages = self._compute_group_relative_advantages(rewards)
        else:
            raise ValueError(f"Unsupported reward dimension: {rewards_tensor.dim()}, expected 1 or 2.")

        return {
            'input_ids': input_ids.detach(),
            'attention_mask': attention_mask.detach(),
            'loss_mask': loss_mask.detach(),
            'old_log_probs': old_log_probs.detach() if old_log_probs is not None else None,
            'ref_log_probs': ref_log_probs.detach() if ref_log_probs is not None else None,
            'completion_ids': completion_ids.detach(),
            'advantages': advantages.detach(),
            'rewards': rewards.detach(),
            'max_new_tokens': max_new_tokens,
        }

    def _grpo_learning_phase(self, rollout_data: dict):
        grpo_batch_size = self.grpo_config.grpo_batch_size
        device = TrainerTools().parallel.device

        input_ids = rollout_data['input_ids']
        attention_mask = rollout_data['attention_mask']
        loss_mask = rollout_data['loss_mask']
        old_log_probs = rollout_data['old_log_probs']
        ref_log_probs = rollout_data['ref_log_probs']
        completion_ids = rollout_data['completion_ids']
        advantages = rollout_data['advantages']

        total_samples = input_ids.shape[0]

        # 若 Rollout 阶段跳过了 old_log_probs 计算，在 multi-epoch (grpo_epochs > 1) 情况下，
        # 仅当单批样本数未超过单个梯度累积窗口 (保证 epoch 0 内部模型权重无中途更新漂移) 时，
        # 记录初始 log_probs 以供后续 epoch 准确作为 old_log_probs 复用
        samples_per_accum_window = grpo_batch_size * self.gradient_accumulation_steps
        if old_log_probs is None and self.grpo_config.grpo_epochs > 1 and total_samples <= samples_per_accum_window:
            cached_old_log_probs = torch.empty(
                (total_samples, completion_ids.shape[1]),
                dtype=torch.float32,
                device=device
            )
        else:
            cached_old_log_probs = None

        valid_rewards = rollout_data['rewards'].float()
        rewards_mean = torch.nanmean(valid_rewards).item() if not torch.isnan(valid_rewards).all() else 0.0

        grpo_stats = {
            "loss": 0.0,
            "moe_aux_loss": 0.0,
            "approx_kl": 0.0,
            "ref_kl": 0.0,
            "clip_frac": 0.0,
            "mean_nll": 0.0,
            "completion_len": 0.0,
            "rewards": 0.0 if math.isnan(rewards_mean) else rewards_mean,
        }

        total_micro_batches_processed = 0

        use_global_token_norm = (
                self.grpo_config.loss_type in ["bnpo", "cispo", "dapo", "vespo"]
                and self.grpo_config.token_level_loss_norm == "global"
        )
        accum_steps = self.gradient_accumulation_steps
        samples_per_accum_window = grpo_batch_size * accum_steps
        num_accum_windows = max((total_samples + samples_per_accum_window - 1) // samples_per_accum_window, 1)

        for grpo_epoch in range(self.grpo_config.grpo_epochs):
            indices = torch.randperm(total_samples, device=device)
            micro_batch_idx = 0

            if use_global_token_norm:
                local_window_tokens = torch.zeros(num_accum_windows, device=device, dtype=torch.float32)
                window_divisors = torch.ones(num_accum_windows, device=device, dtype=torch.float32)
                world_size = TrainerTools().parallel.world_size
                for w in range(num_accum_windows):
                    w_start = w * samples_per_accum_window
                    w_end = min(w_start + samples_per_accum_window, total_samples)
                    w_indices = indices[w_start:w_end]
                    local_window_tokens[w] = loss_mask[w_indices].sum().float()
                    w_accum_steps = max(math.ceil((w_end - w_start) / grpo_batch_size), 1)
                    window_divisors[w] = max(float(world_size * w_accum_steps), 1.0)

                if TrainerTools().parallel.parallel_train and dist.is_initialized():
                    dist.all_reduce(local_window_tokens, op=dist.ReduceOp.SUM)

                # 先 clamp(min=1.0) 保证总 Token 非零，然后再除以各窗口实际的 effective_divisor
                window_norm_factors = local_window_tokens.clamp(min=1.0) / window_divisors
            else:
                window_norm_factors = None

            for i in range(0, total_samples, grpo_batch_size):
                window_idx = min(i // samples_per_accum_window, num_accum_windows - 1)
                avg_tokens_per_mb = (
                    window_norm_factors[window_idx].item()
                    if window_norm_factors is not None
                    else None
                )
                mini_batch_indices = indices[i:i + grpo_batch_size]

                mb_input_ids = input_ids[mini_batch_indices]
                mb_attention_mask = attention_mask[mini_batch_indices]
                mb_completion_ids = completion_ids[mini_batch_indices]
                mb_loss_mask = loss_mask[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices] if old_log_probs is not None else None
                mb_ref_log_probs = ref_log_probs[mini_batch_indices] if ref_log_probs is not None else None
                mb_advantages = advantages[mini_batch_indices]

                actual_completion_len = mb_loss_mask.sum(dim=-1).float().mean().item()
                if self.grpo_config.loss_type == "dr_grpo":
                    completion_len_for_loss = (
                            self.grpo_config.dr_grpo_max_completion_len
                            or self.grpo_config.generate_config.max_seq_len
                    )
                else:
                    completion_len_for_loss = actual_completion_len

                with autocast(TrainerTools().parallel.device_type):
                    log_probs, aux_loss = self._compute_completion_log_probs(
                        self.train_model, mb_input_ids, mb_attention_mask, mb_completion_ids,
                        completion_mask=mb_loss_mask
                    )

                    # 核心复用：若 mb_old_log_probs 为 None（即首轮迭代且跳过了 Rollout 前向），
                    # 直接就地 detach 复用本次带梯度的前向 log_probs
                    if mb_old_log_probs is None:
                        mb_old_log_probs = log_probs.detach()
                        if cached_old_log_probs is not None:
                            cached_old_log_probs[mini_batch_indices] = log_probs.detach().float()

                    loss, clip_frac = self.criterion(
                        log_probs=log_probs,
                        old_log_probs=mb_old_log_probs,
                        ref_log_probs=mb_ref_log_probs,
                        completion_mask=mb_loss_mask,
                        advantages=mb_advantages,
                        completion_len=completion_len_for_loss,
                        avg_tokens_per_micro_batch=avg_tokens_per_mb
                    )

                    with torch.no_grad():
                        fp32_log_probs = log_probs.float()
                        fp32_mask = mb_loss_mask.float()

                        mean_nll = -(fp32_log_probs * fp32_mask).sum() / fp32_mask.sum().clamp(min=1.0)
                        completion_len = fp32_mask.sum(dim=-1).mean()

                    if aux_loss is not None:
                        aux_loss = aux_loss.to(loss.dtype)
                    else:
                        aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                    with torch.no_grad():
                        step_log_ratio = fp32_log_probs - mb_old_log_probs.float()
                        step_approx_kl = (torch.exp(step_log_ratio) - step_log_ratio - 1)
                        approx_kl = (step_approx_kl * fp32_mask).sum() / fp32_mask.sum().clamp(min=1.0)

                        if mb_ref_log_probs is not None:
                            ref_log_ratio = mb_ref_log_probs.float() - fp32_log_probs
                            ref_kl = (torch.exp(ref_log_ratio) - ref_log_ratio - 1)
                            ref_kl = (ref_kl * fp32_mask).sum() / fp32_mask.sum().clamp(min=1.0)
                        else:
                            ref_kl = torch.tensor(0.0, device=loss.device)

                grpo_loss_unscaled = loss + aux_loss

                # 每个 grpo_epoch 的最后一个 micro-batch 强制作为边界提交累积梯度，防止跨 epoch 泄漏
                is_last_mini_batch = (i + grpo_batch_size >= total_samples)
                total_mbs = math.ceil(total_samples / grpo_batch_size)
                mb_idx_0 = i // grpo_batch_size
                window_start_mb = (mb_idx_0 // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                actual_accum_steps = min(self.gradient_accumulation_steps, total_mbs - window_start_mb)

                if self.is_ds:
                    if is_last_mini_batch:
                        self.train_model.set_gradient_accumulation_boundary(True)
                        need_update_step = True
                    else:
                        need_update_step = self.train_model.is_gradient_accumulation_boundary()

                    if actual_accum_steps < self.gradient_accumulation_steps:
                        scale_factor = self.gradient_accumulation_steps / actual_accum_steps
                        self._backward_loss(grpo_loss_unscaled * scale_factor, actual_accum_steps)
                    else:
                        self._backward_loss(grpo_loss_unscaled, self.gradient_accumulation_steps)
                else:
                    micro_batch_idx += 1
                    need_update_step = (
                            micro_batch_idx % self.gradient_accumulation_steps == 0
                            or is_last_mini_batch
                    )
                    self._backward_loss(grpo_loss_unscaled, actual_accum_steps)

                grpo_stats["loss"] += grpo_loss_unscaled.detach().item()
                grpo_stats["moe_aux_loss"] += aux_loss.detach().item()
                grpo_stats["approx_kl"] += approx_kl.detach().item()
                grpo_stats["ref_kl"] += ref_kl.detach().item()
                grpo_stats["clip_frac"] += clip_frac.detach().item()
                grpo_stats["mean_nll"] += mean_nll.detach().item()
                grpo_stats["completion_len"] += completion_len.detach().item()
                total_micro_batches_processed += 1

                if need_update_step:
                    self._update_step(is_last_step=is_last_mini_batch)

            if cached_old_log_probs is not None and old_log_probs is None:
                old_log_probs = cached_old_log_probs


        if total_micro_batches_processed > 0:
            for key in ["loss", "moe_aux_loss", "approx_kl", "ref_kl", "clip_frac", "mean_nll", "completion_len"]:
                grpo_stats[key] /= total_micro_batches_processed

        return grpo_stats

    def train(self):
        global_steps_since_last_save = 0
        global_steps_since_last_eval = 0

        micro_batches_per_epoch = math.ceil((self.train_config.batch_size * self.grpo_config.group_size) / self.grpo_config.grpo_batch_size)
        updates_per_epoch = math.ceil(micro_batches_per_epoch / self.gradient_accumulation_steps)
        updates_per_rollout = self.grpo_config.grpo_epochs * updates_per_epoch
        global_steps_per_rollout = max(1, updates_per_rollout)

        for epoch in range(self.resume_epoch, self.train_config.n_epochs):
            self.train_model.train()
            file_count = len(self.train_config.file_dataset)
            start_file_idx = self.resume_file_idx if epoch == self.resume_epoch else 0

            for file_idx in range(start_file_idx, file_count):
                dataset, file_path = self._create_dataset(file_idx)

                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                batch_count_per_file = len(train_data_loader)
                TrainerTools().parallel.on_epoch_start(epoch, file_idx)
                self._on_file_start(epoch, file_path)

                skip_batches = 0
                if epoch == self.resume_epoch and file_idx == self.resume_file_idx:
                    skip_batches = self.resume_batch_idx
                    if skip_batches > 0 and TrainerTools().parallel.is_main_process:
                        Logger.std_log(f"Fast forwarding {skip_batches} batches in {file_path}...")

                if skip_batches > 0 and self.resume_dataloader_rng_state is not None:
                    torch.set_rng_state(self.resume_dataloader_rng_state.cpu().to(torch.uint8))
                    self.resume_dataloader_rng_state = None

                dataloader_rng_state = torch.get_rng_state()
                data_iterator = iter(train_data_loader)
                if skip_batches > 0:
                    data_iterator = islice(data_iterator, skip_batches, None)

                for batch, batch_data in enumerate(data_iterator):
                    if self.resume_rng_state is not None:
                        restore_rng_state_dict(self.resume_rng_state)
                        self.resume_rng_state = None

                    batch = skip_batches + batch

                    # start generate
                    if TrainerTools().parallel.is_main_process:
                        Logger.std_log(f'start generate for batch {batch + 1}/{batch_count_per_file}')

                    # 生成数据 (阶段一：采样生成)
                    sample_data = None
                    sample_error = None

                    try:
                        sample_data = self._sample_rollout_completions(batch_data)
                    except Exception as e:
                        sample_error = e

                    # 第一阶段跨卡同步：确保所有 Rank 采样生成均成功完成，防止部分卡崩溃引发后续 ZeRO-3 集合通信死锁
                    if TrainerTools().parallel.parallel_train:
                        has_sample_err = torch.tensor(1 if sample_error is not None else 0, device=TrainerTools().parallel.device, dtype=torch.int32)
                        dist.all_reduce(has_sample_err, op=dist.ReduceOp.MAX)
                        if has_sample_err.item() > 0:
                            if sample_error is not None:
                                self._on_exception(sample_error, epoch, batch)
                            else:
                                if TrainerTools().parallel.is_main_process:
                                    Logger.std_log(f"WARN: Rollout sampling failed on a peer rank in epoch {epoch}, batch {batch + 1}. Aborting to prevent collective deadlock.")
                                raise RuntimeError(f"Rollout sampling failed on a peer rank in epoch {epoch}, batch {batch + 1}.")
                    elif sample_error is not None:
                        self._on_exception(sample_error, epoch, batch)

                    # 评估数据 (阶段二：计算 LogProbs 与 奖励优势)
                    rollout_data = None
                    eval_error = None
                    try:
                        rollout_data = self._evaluate_rollout_data(sample_data)
                    except Exception as e:
                        eval_error = e

                    # 第二阶段跨卡同步：确保所有 Rank 评估打分均成功完成
                    if TrainerTools().parallel.parallel_train:
                        has_eval_err = torch.tensor(1 if eval_error is not None else 0, device=TrainerTools().parallel.device, dtype=torch.int32)
                        dist.all_reduce(has_eval_err, op=dist.ReduceOp.MAX)
                        if has_eval_err.item() > 0:
                            if eval_error is not None:
                                self._on_exception(eval_error, epoch, batch)
                            else:
                                if TrainerTools().parallel.is_main_process:
                                    Logger.std_log(f"WARN: Rollout evaluation failed on a peer rank in epoch {epoch}, batch {batch + 1}. Aborting to prevent collective deadlock.")
                                raise RuntimeError(f"Rollout evaluation failed on a peer rank in epoch {epoch}, batch {batch + 1}.")
                    elif eval_error is not None:
                        self._on_exception(eval_error, epoch, batch)
                    # end generate

                    try:
                        if TrainerTools().parallel.is_main_process:
                            Logger.std_log(f'start train for batch {batch + 1}/{batch_count_per_file}')

                        grpo_stats = self._grpo_learning_phase(rollout_data)
                        global_steps_since_last_save += global_steps_per_rollout
                        global_steps_since_last_eval += global_steps_per_rollout

                        stats_tensor = torch.tensor([
                            grpo_stats['loss'],
                            grpo_stats['moe_aux_loss'],
                            grpo_stats['approx_kl'],
                            grpo_stats['ref_kl'],
                            grpo_stats['clip_frac'],
                            grpo_stats['mean_nll'],
                            grpo_stats['rewards'],
                            grpo_stats['completion_len']
                        ], device=TrainerTools().parallel.device)

                        if TrainerTools().parallel.parallel_train:
                            if TrainerTools().parallel.device_type == 'mlu':
                                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                                stats_tensor.div_(TrainerTools().parallel.world_size)
                            else:
                                dist.all_reduce(stats_tensor, dist.ReduceOp.AVG)

                        self._log(
                            keys={
                                'epoch': epoch,
                                'file': f'{file_idx + 1}/{file_count}',
                                'batch': f'{batch + 1}/{batch_count_per_file}',
                            },
                            values={
                                'loss/total': stats_tensor[0].item(),
                                'loss/moe_aux': stats_tensor[1].item(),
                                'rl/approx_kl': stats_tensor[2].item(),
                                'rl/ref_kl': stats_tensor[3].item(),
                                'rl/clip_frac': stats_tensor[4].item(),
                                'rl/mean_nll': stats_tensor[5].item(),
                                'env/reward_total': stats_tensor[6].item(),
                                'env/completion_len': stats_tensor[7].item()
                            }
                        )

                        if 0 < self.train_config.save_interval <= global_steps_since_last_save:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            save_steps(
                                epoch=epoch,
                                file_idx=file_idx,
                                batch_idx=batch + 1,
                                lr_scheduler=self.lr_scheduler,
                                dataloader_rng_state=dataloader_rng_state
                            )
                            global_steps_since_last_save %= self.train_config.save_interval

                        if 0 < self.train_config.eval_interval <= global_steps_since_last_eval:
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                            global_steps_since_last_eval %= self.train_config.eval_interval
                    except Exception as e:
                        self._on_exception(e, epoch, batch)

                if self.resume_rng_state is not None:
                    restore_rng_state_dict(self.resume_rng_state)
                    self.resume_rng_state = None

                try:
                    # 一个文件训练结束后，清理内存
                    del train_data_loader
                    del dataset
                    del data_iterator
                    del dataloader_rng_state
                    del rollout_data
                    del batch_data
                except UnboundLocalError: ...

                if hasattr(TrainerTools().parallel, '_sampler'):
                    TrainerTools().parallel._sampler = None

                gc.collect()
                empty_cache()

            # end epoch
            # reset resume state
            self.resume_file_idx = 0
            self.resume_batch_idx = 0

            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
            save_steps(
                epoch=epoch + 1,
                file_idx=0,
                batch_idx=0,
                lr_scheduler=self.lr_scheduler
            )

            TrainerTools().parallel.on_epoch_end(epoch)
            self._on_epoch_end(tag=f'epoch:{epoch}')

        TrainerTools().parallel.destroy()