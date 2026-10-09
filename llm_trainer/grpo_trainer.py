from typing import Tuple, List, Optional
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
    fold_rewards_to_preceding_action
)
from .checkpoint import save_checkpoint, save_steps
from .partition_utils import unwrap_model_for_generation


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
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            generation_service=generation_service,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps
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
            chunk_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算 Completion 部分的 LogProbs：
        1. 采用 Active Token Filtering：跳过 Prompt 的 lm_head 矩阵乘法
        2. 支持 chunk_size 配置：None 为全量计算，整数则分块计算
        """
        def _step_forward(inp_ids, att_mask, comp_ids):
            pos_ids = calc_position_ids(att_mask)
            comp_len = comp_ids.size(1)
            outputs = model(
                input_ids=inp_ids,
                attention_mask=att_mask,
                position_ids=pos_ids,
                return_logits=True,
                logits_to_keep=comp_len + 1
            )

            logits_completion = outputs['logits'][:, :-1, :][:, -comp_len:, :].float()
            log_probs = log_softmax(logits_completion, comp_ids)
            aux_loss = outputs['aux_loss']

            return log_probs, aux_loss

        if chunk_size is None or chunk_size <= 0 or chunk_size >= input_ids.size(0):
            return _step_forward(input_ids, attention_mask, completion_ids)

        all_log_probs = []
        aux_losses = []
        for i in range(0, input_ids.size(0), chunk_size):
            c_input_ids = input_ids[i: i + chunk_size]
            c_attention_mask = attention_mask[i: i + chunk_size]
            c_completion_ids = completion_ids[i: i + chunk_size]
            log_probs, aux_loss = _step_forward(c_input_ids, c_attention_mask, c_completion_ids)
            all_log_probs.append(log_probs)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        total_log_probs = torch.cat(all_log_probs, dim=0)
        total_aux_loss = sum(aux_losses) / len(aux_losses) if len(aux_losses) > 0 else None

        return total_log_probs, total_aux_loss

    def _compute_group_relative_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        group_size = self.grpo_config.group_size
        rewards_fp32 = torch.nan_to_num(rewards.float(), nan=0.0)

        # Reshape rewards to group by prompt: [batch, group_size]
        rewards_by_group = rewards_fp32.view(-1, group_size)

        # Compute mean and standard deviation for each prompt group: [batch]
        group_means = rewards_by_group.mean(dim=1)
        expanded_means = group_means.repeat_interleave(group_size)

        # 判断是否进行标准差 (std) 缩放：
        # 若 scale_rewards 为 None，当 loss_type == 'dr_grpo' 时默认设为 False（符合 Dr. GRPO 算法规范），其他模式默认设为 True
        scale_rewards = self.grpo_config.scale_rewards
        if scale_rewards is None:
            scale_rewards = (self.grpo_config.loss_type != "dr_grpo")

        if scale_rewards:
            group_stds = torch.nan_to_num(rewards_by_group.std(dim=1, unbiased=False), nan=0.0)
            expanded_stds = group_stds.repeat_interleave(group_size)

            is_flat = expanded_stds < 1e-6
            norm_adv = (rewards_fp32 - expanded_means) / (expanded_stds + 1e-4)
            advantages = torch.where(is_flat, torch.zeros_like(norm_adv), norm_adv)
        else:
            # Dr. GRPO 核心改动：去掉组内 std 归一化，仅做均值中心化
            advantages = rewards_fp32 - expanded_means

        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        return advantages.to(dtype=rewards.dtype).unsqueeze(1)

    def _compute_multi_turn_advantages(
            self,
            rewards_tensor: torch.Tensor,
            loss_mask: torch.Tensor,
            gamma: float = 1.0
    ) -> torch.Tensor:
        """
        针对多轮交互 Agent 的因果时序信用分配 (Causal Reward-to-Go + Action-level Bridge Discount)。

        - 轮内 Token: 属于同一次模型输出，共享本轮及未来的折现回报，轮内不额外衰减；
        - 跨轮跃迁: 越过环境 Feedback 区间时乘以跨轮折扣 gamma；
        - 组内对齐: 基于同 Prompt 组且在同一交互轮次内对齐计算基线均值与标准差，彻底消除跨轮时序负偏移偏差。
        """
        batch_size, seq_len = rewards_tensor.shape
        group_size = self.grpo_config.group_size
        device = rewards_tensor.device

        # 1. 识别每个 Token 所属的 Action 交互轮次 (Turn ID)
        # Action 区间为 loss_mask == True 的连续段，每个连续段对应一次模型生成的交互轮次
        is_turn_start = loss_mask & torch.cat(
            [loss_mask.new_ones(batch_size, 1), ~loss_mask[:, :-1]],
            dim=1
        )
        turn_ids = torch.cumsum(is_turn_start.long(), dim=1)  # [batch_size, seq_len]
        max_turns = turn_ids.max().item()

        # 若全序列无任何有效生成 Token，直接返回全零
        if max_turns == 0:
            return torch.zeros_like(rewards_tensor)

        # 2. 统计各样本在每个交互轮次所获得的有效奖励 (Turn Rewards)
        # 屏蔽非生成位置（Feedback / Pad）的噪声
        masked_rewards = (rewards_tensor * loss_mask.float()).float()
        valid_turn_ids = torch.where(loss_mask, turn_ids, torch.zeros_like(turn_ids))

        turn_rewards = torch.zeros(batch_size, max_turns + 1, device=device, dtype=torch.float32)
        turn_rewards.scatter_add_(dim=1, index=valid_turn_ids, src=masked_rewards)

        # 标记各样本在每个交互轮次是否实际存在模型 Action（解决异构轮次下的虚假 0 统计偏差）
        turn_active = torch.zeros(batch_size, max_turns + 1, dtype=torch.bool, device=device)
        turn_active.scatter_(dim=1, index=valid_turn_ids, value=True)
        turn_active = turn_active[:, 1:]  # [batch_size, max_turns]

        # 3. 按因果时序反向累积计算各轮次的未来折现回报 (Turn-level Return-to-Go)
        # G_{i, k} = R_{i, k} + gamma * G_{i, k+1}
        turn_returns = torch.zeros(batch_size, max_turns + 1, device=device, dtype=torch.float32)
        running_return = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for k in range(max_turns, 0, -1):
            running_return = turn_rewards[:, k] + gamma * running_return
            turn_returns[:, k] = running_return

        # 4. 组内对齐基线：严格仅在各组内【实际发生交互的活跃样本】之间计算均值与方差
        # 提取有效轮次 1..max_turns，形状为 [num_groups, group_size, max_turns]
        returns_by_group = turn_returns[:, 1:].view(-1, group_size, max_turns)
        active_by_group = turn_active.view(-1, group_size, max_turns)
        active_mask = active_by_group.float()
        active_counts = active_mask.sum(dim=1, keepdim=True)  # [num_groups, 1, max_turns]

        # 仅对同组内活跃参与该轮交互的样本计算基线均值
        masked_returns = returns_by_group * active_mask
        group_means = masked_returns.sum(dim=1, keepdim=True) / active_counts.clamp(min=1.0)

        # 仅对同组内活跃样本计算方差与标准差
        diff_sq = ((returns_by_group - group_means) ** 2) * active_mask
        var = diff_sq.sum(dim=1, keepdim=True) / active_counts.clamp(min=1.0)
        group_stds = torch.sqrt(var)

        # 整组全轨迹总回报基线（第 1 轮累积总回报，包含同组全部 group_size 个候选），用于单活跃样本的基线借调
        overall_returns = turn_returns[:, 1].view(-1, group_size, 1)  # [num_groups, group_size, 1]
        overall_means = overall_returns.mean(dim=1, keepdim=True)      # [num_groups, 1, 1]
        overall_stds = torch.nan_to_num(overall_returns.std(dim=1, keepdim=True, unbiased=False), nan=0.0)

        scale_rewards = self.grpo_config.scale_rewards
        if scale_rewards is None:
            scale_rewards = (self.grpo_config.loss_type != "dr_grpo")

        if scale_rewards:
            # 1. 多候选活跃（active_counts >= 2）：同轮次活跃候选内部标准化
            valid_multi = (active_counts >= 2) & (group_stds > 1e-6)
            norm_adv_multi = (returns_by_group - group_means) / (group_stds + 1e-4)
            adv_multi = torch.where(valid_multi, norm_adv_multi, torch.zeros_like(returns_by_group))

            # 2. 单候选活跃（active_counts == 1）：借调整组全轨迹相对优势 (Full-trajectory Relative Advantage)，
            # 消除 G_{i,k} 残余回报天然小于全轨迹总回报均值 E[G_1] 所造成的负向尺度错位偏差
            valid_single = (active_counts == 1) & (overall_stds > 1e-6)
            overall_adv = (overall_returns - overall_means) / (overall_stds + 1e-4)
            adv_single = torch.where(valid_single, overall_adv.expand_as(returns_by_group), torch.zeros_like(returns_by_group))

            turn_adv = torch.where(active_counts >= 2, adv_multi, adv_single)
        else:
            # Dr. GRPO 模式（仅中心化，不除以标准差）
            adv_multi = torch.where(active_counts >= 2, returns_by_group - group_means, torch.zeros_like(returns_by_group))
            overall_adv = overall_returns - overall_means
            adv_single = torch.where(active_counts == 1, overall_adv.expand_as(returns_by_group), torch.zeros_like(returns_by_group))
            turn_adv = torch.where(active_counts >= 2, adv_multi, adv_single)

        # 严格将非活跃轮次的优势值置零
        turn_adv = turn_adv * active_mask

        # 5. 将各轮次标准化优势值精确广播映射回对应 Action Token
        turn_adv = turn_adv.view(batch_size, max_turns)
        turn_adv_padded = torch.cat(
            [torch.zeros(batch_size, 1, device=device, dtype=turn_adv.dtype), turn_adv],
            dim=1
        )

        advantages = torch.gather(turn_adv_padded, dim=1, index=turn_ids)
        advantages = torch.where(loss_mask, advantages, torch.zeros_like(advantages))
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        return advantages.to(dtype=rewards_tensor.dtype)

    def _compute_token_level_prm_advantages(
            self,
            rewards_tensor: torch.Tensor,
            loss_mask: torch.Tensor,
            gamma: float = 1.0
    ) -> torch.Tensor:
        """
        针对单轮推理任务中 2D Token-level / Step-level PRM 过程奖励的细粒度因果时序信用分配。

        1. 沿时序从后向前反向累积计算各 Token 的因果未来折现回报 (Reward-to-Go):
           G_{i, t} = r_{i, t} + gamma * G_{i, t+1}
        2. 按同 Prompt 组在各时步活跃 Token 之间进行组内归一化：
           - 多候选活跃 (active_counts >= 2): 同步活跃候选内部标准化；
           - 单候选活跃 (active_counts == 1): 借调整体全轨迹相对优势；
        3. 保留 Token-level 细粒度优势分布，彻底解决 PRM 奖励被粗暴求和退化为标量 ORM 的缺陷。
        """
        batch_size, seq_len = rewards_tensor.shape
        group_size = self.grpo_config.group_size
        device = rewards_tensor.device

        masked_rewards = (rewards_tensor * loss_mask.float()).float()

        # 1. 反向累积因果 Reward-to-Go
        r2g = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
        running_r2g = torch.zeros(batch_size, device=device, dtype=torch.float32)
        for t in range(seq_len - 1, -1, -1):
            running_r2g = masked_rewards[:, t] + gamma * running_r2g * loss_mask[:, t].float()
            r2g[:, t] = running_r2g * loss_mask[:, t].float()

        # 2. 组内归一化
        r2g_by_group = r2g.view(-1, group_size, seq_len)
        mask_by_group = loss_mask.view(-1, group_size, seq_len).float()
        active_counts = mask_by_group.sum(dim=1, keepdim=True)  # [num_groups, 1, seq_len]

        masked_r2g = r2g_by_group * mask_by_group
        group_means = masked_r2g.sum(dim=1, keepdim=True) / active_counts.clamp(min=1.0)
        diff_sq = ((r2g_by_group - group_means) ** 2) * mask_by_group
        group_vars = diff_sq.sum(dim=1, keepdim=True) / active_counts.clamp(min=1.0)
        group_stds = torch.sqrt(group_vars)

        # 提取各候选在首个有效 Token 处的全轨迹累积总回报（用于单活跃候选借调）
        first_valid_idx = loss_mask.int().argmax(dim=1, keepdim=True)
        overall_returns = r2g.gather(dim=1, index=first_valid_idx).view(-1, group_size, 1)
        overall_means = overall_returns.mean(dim=1, keepdim=True)
        overall_stds = torch.nan_to_num(overall_returns.std(dim=1, keepdim=True, unbiased=False), nan=0.0)

        scale_rewards = self.grpo_config.scale_rewards
        if scale_rewards is None:
            scale_rewards = (self.grpo_config.loss_type != "dr_grpo")

        if scale_rewards:
            valid_multi = (active_counts >= 2) & (group_stds > 1e-6)
            norm_adv_multi = (r2g_by_group - group_means) / (group_stds + 1e-4)
            adv_multi = torch.where(valid_multi, norm_adv_multi, torch.zeros_like(r2g_by_group))

            valid_single = (active_counts == 1) & (overall_stds > 1e-6)
            overall_adv = (overall_returns - overall_means) / (overall_stds + 1e-4)
            adv_single = torch.where(valid_single, overall_adv.expand_as(r2g_by_group), torch.zeros_like(r2g_by_group))

            advantages = torch.where(active_counts >= 2, adv_multi, adv_single)
        else:
            adv_multi = torch.where(active_counts >= 2, r2g_by_group - group_means, torch.zeros_like(r2g_by_group))
            overall_adv = overall_returns - overall_means
            adv_single = torch.where(active_counts == 1, overall_adv.expand_as(r2g_by_group), torch.zeros_like(r2g_by_group))
            advantages = torch.where(active_counts >= 2, adv_multi, adv_single)

        advantages = (advantages * mask_by_group).view(batch_size, seq_len)
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
            else:
                with unwrap_model_for_generation(self.train_model) as unwrapped_model:
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

        completion_pad_mask = completion_ids != pad_token_id

        if external_gen_mask is not None:
            loss_mask = completion_pad_mask & external_gen_mask
        else:
            loss_mask = completion_pad_mask

        input_ids = torch.cat([padded_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_masks, completion_pad_mask], dim=1)

        chunk_size = self.grpo_config.chunked_log_probs_size
        with torch.no_grad():
            with autocast(TrainerTools().parallel.device_type):
                old_log_probs, _ = self._compute_completion_log_probs(
                    self.train_model, input_ids, attention_mask, completion_ids, chunk_size
                )

            if self.ref_model:
                with autocast(TrainerTools().parallel.device_type):
                    ref_log_probs, _ = self._compute_completion_log_probs(
                        self.ref_model, input_ids, attention_mask, completion_ids, chunk_size
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

        if getattr(self.grpo_config, 'missing_eos_penalty', None) is not None:
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
                # 多轮 RL 模式下接收 1D 轨迹标量奖励：精准注入到各样本最后一个有效 Action Token 上，
                # 随后通过 _compute_multi_turn_advantages 沿历史轮次反向折现传递 (Reward-to-Go)，
                # 彻底消除多轮 1D 奖励退化为单轮标量处理的旁路缺陷
                dense_rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
                if valid_indices_mask.any():
                    valid_batch = batch_range[valid_indices_mask]
                    valid_last = last_token_indices[valid_indices_mask]
                    dense_rewards[valid_batch, valid_last] = rewards_tensor[valid_indices_mask]
                rewards = rewards_tensor
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
                dense_rewards = rewards_tensor
                masked_rewards = dense_rewards * loss_mask.float()
                rewards = masked_rewards.sum(dim=-1)

            # 多轮因果时序信用分配 (Reward-to-Go + Action-level Bridge Discount)
            advantages = self._compute_multi_turn_advantages(
                rewards_tensor=dense_rewards,
                loss_mask=loss_mask,
                gamma=getattr(self.grpo_config, 'gamma', 1.0)
            )
        elif rewards_tensor.dim() == 2:
            # 单轮模式 2D 过程奖励 (Token-level PRM dense reward)：
            # 严格沿时序计算 Token-level 因果累积未来回报 (Reward-to-Go)，
            # 并在同组候选的对应 Token 位置进行组内归一化，实现细粒度 Token-level 信用分配
            assert rewards_tensor.shape == completion_ids.shape, (
                f"2D dense reward shape {rewards_tensor.shape} must match completion_ids shape {completion_ids.shape}"
            )
            dense_rewards = rewards_tensor
            masked_rewards = dense_rewards * loss_mask.float()
            rewards = masked_rewards.sum(dim=-1)

            advantages = self._compute_token_level_prm_advantages(
                rewards_tensor=dense_rewards,
                loss_mask=loss_mask,
                gamma=getattr(self.grpo_config, 'gamma', 1.0)
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
            'old_log_probs': old_log_probs.detach(),
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

        grpo_stats = {
            "loss": 0.0,
            "moe_aux_loss": 0.0,
            "approx_kl": 0.0,
            "ref_kl": 0.0,
            "clip_frac": 0.0,
            "mean_nll": 0.0,
            "completion_len": 0.0,
            "rewards": rollout_data['rewards'].mean().item(),
        }

        total_micro_batches_processed = 0
        global_micro_batch_idx = 0

        # 计算全局 Token 级归一化因子（仅用于 DAPO / BNPO / CISPO / VESPO 等消除 micro-batch 划分造成的长度梯度权重偏差）
        # 对于标准 "grpo" 以及 "sapo", "dr_grpo", "luspo"，不依赖全局 token 归一化，跳过跨卡通信
        use_global_token_norm = (
            self.grpo_config.loss_type in ["bnpo", "cispo", "dapo", "vespo"]
            and self.grpo_config.token_level_loss_norm == "global"
        )
        if use_global_token_norm:
            local_total_tokens = loss_mask.sum().float()
            num_micro_batches_local = (total_samples + grpo_batch_size - 1) // grpo_batch_size
            if TrainerTools().parallel.parallel_train and dist.is_initialized():
                global_total_tokens = local_total_tokens.clone()
                dist.all_reduce(global_total_tokens, op=dist.ReduceOp.SUM)
                total_micro_batches_global = num_micro_batches_local * TrainerTools().parallel.world_size
                avg_tokens_per_mb = (global_total_tokens / max(total_micro_batches_global, 1)).item()
            else:
                avg_tokens_per_mb = (local_total_tokens / max(num_micro_batches_local, 1)).item()
            avg_tokens_per_mb = max(avg_tokens_per_mb, 1.0)
        else:
            avg_tokens_per_mb = None

        for grpo_epoch in range(self.grpo_config.grpo_epochs):
            indices = torch.randperm(total_samples, device=device)

            for i in range(0, total_samples, grpo_batch_size):
                mini_batch_indices = indices[i:i + grpo_batch_size]

                mb_input_ids = input_ids[mini_batch_indices]
                mb_attention_mask = attention_mask[mini_batch_indices]
                mb_completion_ids = completion_ids[mini_batch_indices]
                mb_loss_mask = loss_mask[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices]
                mb_ref_log_probs = ref_log_probs[mini_batch_indices] if ref_log_probs is not None else None
                mb_advantages = advantages[mini_batch_indices]

                actual_completion_len = mb_loss_mask.sum(dim=-1).float().mean().item()
                if self.grpo_config.loss_type == "dr_grpo":
                    completion_len_for_loss = (
                        self.grpo_config.dr_grpo_max_completion_len
                        or rollout_data.get('max_new_tokens')
                        or mb_completion_ids.size(1)
                    )
                else:
                    completion_len_for_loss = actual_completion_len

                with autocast(TrainerTools().parallel.device_type):
                    log_probs, aux_loss = self._compute_completion_log_probs(
                        self.train_model, mb_input_ids, mb_attention_mask, mb_completion_ids
                    )

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

                is_last_mini_batch = (
                    grpo_epoch == self.grpo_config.grpo_epochs - 1
                    and (i + grpo_batch_size >= total_samples)
                )

                if self.is_ds:
                    if is_last_mini_batch:
                        self.train_model.set_gradient_accumulation_boundary(True)
                        need_update_step = True
                    else:
                        need_update_step = self.train_model.is_gradient_accumulation_boundary()
                else:
                    global_micro_batch_idx += 1
                    need_update_step = (
                        global_micro_batch_idx % self.gradient_accumulation_steps == 0
                        or is_last_mini_batch
                    )

                self._backward_loss(grpo_loss_unscaled, self.gradient_accumulation_steps)

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


        if total_micro_batches_processed > 0:
            for key in ["loss", "moe_aux_loss", "approx_kl", "ref_kl", "clip_frac", "mean_nll", "completion_len"]:
                grpo_stats[key] /= total_micro_batches_processed

        return grpo_stats

    def train(self):
        global_steps_since_last_save = 0
        global_steps_since_last_eval = 0

        micro_batches_per_rollout = (self.train_config.batch_size * self.grpo_config.group_size) / self.grpo_config.grpo_batch_size
        updates_per_rollout = (self.grpo_config.grpo_epochs * micro_batches_per_rollout) / self.gradient_accumulation_steps
        global_steps_per_rollout = max(1, math.ceil(updates_per_rollout))

        for epoch in range(self.resume_epoch, self.train_config.n_epochs):
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
                TrainerTools().parallel.on_epoch_start(epoch)
                self._on_file_start(epoch, file_path)

                skip_batches = 0
                if epoch == self.resume_epoch and file_idx == self.resume_file_idx:
                    skip_batches = self.resume_batch_idx
                    if skip_batches > 0 and TrainerTools().parallel.is_main_process:
                        Logger.std_log(f"Fast forwarding {skip_batches} batches in {file_path}...")

                data_iterator = iter(train_data_loader)
                if skip_batches > 0:
                    data_iterator = islice(data_iterator, skip_batches, None)

                for batch, batch_data in enumerate(data_iterator):
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
                                lr_scheduler=self.lr_scheduler
                            )
                            global_steps_since_last_save %= self.train_config.save_interval

                        if 0 < self.train_config.eval_interval <= global_steps_since_last_eval:
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                            global_steps_since_last_eval %= self.train_config.eval_interval
                    except Exception as e:
                        self._on_exception(e, epoch, batch)

                try:
                    # 一个文件训练结束后，清理内存
                    del train_data_loader
                    del dataset
                    del data_iterator
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