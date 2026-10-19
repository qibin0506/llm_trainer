from abc import ABC, abstractmethod
from typing import List, Optional
import math

import torch

from .log import Logger

class LRScheduler(ABC):
    @property
    @abstractmethod
    def cur_steps(self): ...

    @property
    @abstractmethod
    def cur_lr(self): ...

    @abstractmethod
    def step(self): ...

    @abstractmethod
    def get_ckpt_dict(self) -> dict: ...

    @abstractmethod
    def restore_ckpt_dict(self, ckpt: dict): ...


class WarmupCosineAnnealingLRScheduler(LRScheduler):
    def __init__(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            warmup_iters: int,
            initial_lr: float,
            min_lr: Optional[float] = 0.0,
            max_lr: Optional[float] = None,
            cosine_annealing_period: int,  # 每个周期的步数
            cosine_annealing_period_mul: int = 0,  # 周期长度的倍数
            param_group_indices: Optional[List[int]] = None,
            need_log: bool = False
    ):
        super().__init__()

        self._optimizer = optimizer
        self._min_lr = min_lr if min_lr is not None else 0.0
        self._warmup_iters = warmup_iters if warmup_iters is not None else 0

        if max_lr is None:
            self._max_lr = initial_lr
            self._initial_lr = self._min_lr if self._warmup_iters > 0 else initial_lr
        else:
            self._max_lr = max_lr
            self._initial_lr = initial_lr

        self._cosine_annealing_period = cosine_annealing_period
        self._cosine_annealing_period_mul = cosine_annealing_period_mul
        self.param_group_indices = param_group_indices

        self._group_max_lrs = []
        primary_groups = self._get_primary_param_groups()
        for g in primary_groups:
            g_max = g.get('max_lr')
            if g_max is None:
                params = g.get('params', [])
                if g.get('use_muon', False) or any(getattr(p, 'use_muon', False) for p in params):
                    g_max = g.get('lr', self._max_lr)
                else:
                    g_max = self._max_lr
            self._group_max_lrs.append(g_max)

        self.T_cur = 0
        self.cycle = 0

        if self._warmup_iters > 0:
            self._lr_increment = (self._max_lr - self._initial_lr) / self._warmup_iters
        else:
            self._lr_increment = 0.0

        self._steps = 0
        self._current_lr = self._initial_lr
        self._cosine_annealing_base_lr = None

        if need_log:
            self.logger = Logger('lr.txt')
        else:
            self.logger = None

        # 由于训练循环中 optimizer.step() 先于 lr_scheduler.step() 执行，
        # 初始化时直接设置第 1 个优化步应使用的学习率（当 warmup_iters > 0 时为 initial_lr + 1 * lr_increment，而非 0.0）
        self._update_lr()

    def _get_all_param_groups_lists(self) -> List[list]:
        groups_lists = []
        outer_groups = getattr(self._optimizer, 'param_groups', None)
        if outer_groups is not None:
            groups_lists.append(outer_groups)
        inner_opt = getattr(self._optimizer, 'optimizer', None)
        if inner_opt is not None:
            inner_groups = getattr(inner_opt, 'param_groups', None)
            if inner_groups is not None and inner_groups is not outer_groups:
                groups_lists.append(inner_groups)
        return groups_lists

    def _get_primary_param_groups(self) -> list:
        groups_lists = self._get_all_param_groups_lists()
        return groups_lists[0] if groups_lists else []

    @property
    def cur_steps(self):
        return self._steps

    @property
    def cur_lr(self):
        return self._current_lr

    def step(self):
        self._steps += 1
        self._update_lr()

    def _apply_lr(self, lr: float):
        self._current_lr = lr
        ratio = lr / max(self._max_lr, 1e-12)

        for param_groups in self._get_all_param_groups_lists():
            if self.param_group_indices is None:
                target_groups = param_groups
                target_max_lrs = self._group_max_lrs
            else:
                target_groups = [param_groups[i] for i in self.param_group_indices if i < len(param_groups)]
                target_max_lrs = [self._group_max_lrs[i] for i in self.param_group_indices if i < len(self._group_max_lrs)]

            for param_group, base_max_lr in zip(target_groups, target_max_lrs):
                param_group['lr'] = base_max_lr * ratio

        if self.logger:
            self.logger.log(f"step: {self.cur_steps}, lr: {lr}", log_to_console=False)

    def _update_lr(self):
        # 1. Warmup 阶段优先判断：_steps 表示已完成的优化步数，因此下一步 (第 _steps + 1 步) 的预热进度为 (_steps + 1)
        if self._steps < self._warmup_iters:
            lr = self._initial_lr + (self._steps + 1) * self._lr_increment
        elif self._steps == 0 and self._warmup_iters == 0:
            lr = self._initial_lr
        # 2. 未配置余弦退火周期或周期 <= 0 时保持 max_lr
        elif not self._cosine_annealing_period or self._cosine_annealing_period <= 0:
            lr = self._max_lr
        # 3. 超出最大退火步数且无后续周期时保持 min_lr
        elif self._cosine_annealing_period_mul == 0 and self._steps >= self._cosine_annealing_period + self._warmup_iters:
            lr = self._min_lr
        # 4. 余弦退火阶段
        else:
            if self._cosine_annealing_base_lr is None:
                self._cosine_annealing_base_lr = self._max_lr

            T_max = max(1, self._cosine_annealing_period * (max(self._cosine_annealing_period_mul, 1) ** self.cycle))
            self.T_cur += 1
            calc_t = self.T_cur

            if self.T_cur >= T_max:
                if self._cosine_annealing_period_mul == 0:
                    self.T_cur = T_max
                    calc_t = T_max
                else:
                    self.cycle += 1
                    self.T_cur = 0
                    calc_t = T_max
                    self._cosine_annealing_base_lr = self._max_lr

            cos_factor = (1 + math.cos(math.pi * calc_t / T_max)) / 2
            lr = self._min_lr + (self._cosine_annealing_base_lr - self._min_lr) * cos_factor

        self._apply_lr(lr)

    def get_ckpt_dict(self) -> dict:
        return {
            'cur_lr': self._current_lr,
            'lr_steps': self.cur_steps,
            'cosine_annealing_base_lr': self._cosine_annealing_base_lr,
            't_cur': self.T_cur,
            'cycle': self.cycle,
        }

    def restore_ckpt_dict(self, ckpt: dict):
        if 'cur_lr' in ckpt:
            self._current_lr = ckpt['cur_lr']

        if 'lr_steps' in ckpt:
            self._steps = ckpt['lr_steps']

        if 'cosine_annealing_base_lr' in ckpt:
            self._cosine_annealing_base_lr = ckpt['cosine_annealing_base_lr']

        if 't_cur' in ckpt:
            self.T_cur = ckpt['t_cur']

        if 'cycle' in ckpt:
            self.cycle = ckpt['cycle']

        self._apply_lr(self._current_lr)


class NoneLRScheduler(LRScheduler):
    def __init__(self, initial_lr):
        self._current_lr = initial_lr

    @property
    def cur_steps(self):
        return -1

    @property
    def cur_lr(self):
        return self._current_lr

    def step(self): ...

    def get_ckpt_dict(self) -> dict:
        return {'cur_lr': self._current_lr}

    def restore_ckpt_dict(self, ckpt: dict):
        if 'cur_lr' in ckpt:
            self._current_lr = ckpt['cur_lr']


class CompositeLRScheduler(LRScheduler):
    def __init__(self, schedulers: List[LRScheduler]):
        self.schedulers = schedulers

    @property
    def cur_steps(self):
        return self.schedulers[0].cur_steps if self.schedulers else 0

    @property
    def cur_lr(self):
        return self.schedulers[0].cur_lr if self.schedulers else 0.0

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_ckpt_dict(self) -> dict:
        ckpt = {}
        for i, scheduler in enumerate(self.schedulers):
            ckpt[f'scheduler_{i}'] = scheduler.get_ckpt_dict()
        return ckpt

    def restore_ckpt_dict(self, ckpt: dict):
        for i, scheduler in enumerate(self.schedulers):
            key = f'scheduler_{i}'
            if key in ckpt:
                scheduler.restore_ckpt_dict(ckpt[key])