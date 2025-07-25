from abc import ABC, abstractmethod
import math
import torch
from .log import (
    log,
    get_log_dir
)

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
    def can_clip_grad(self): ...

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
            min_lr: float,
            max_lr: float,
            cosine_annealing_period: int, # 每个周期的步数
            cosine_annealing_period_mul: int = 0, # 周期长度的倍数
            need_log: bool = False
    ):
        super().__init__()

        self._optimizer = optimizer
        self._initial_lr = initial_lr
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._warmup_iters = warmup_iters

        self._cosine_annealing_period = cosine_annealing_period
        self._cosine_annealing_period_mul = cosine_annealing_period_mul

        self.T_cur = 0  # 当前周期内已走过的步数
        self.cycle = 0  # 当前周期编号

        if warmup_iters != 0:
            self._lr_increment = (max_lr - initial_lr) / warmup_iters
        else:
            self._lr_increment = 0

        self._steps = -1
        self._current_lr = initial_lr
        self._cosine_annealing_base_lr = None

        self.need_log = need_log


    @property
    def cur_steps(self):
        return self._steps

    @property
    def cur_lr(self):
        return self._current_lr

    def step(self):
        self._steps += 1
        self._update_lr()

    def can_clip_grad(self):
        return self._steps > self._warmup_iters

    def _update_lr(self):
        # 如果period_mul是0，则认为没有周期，超过余弦退火总步数，则一直保持最小lr
        if self._cosine_annealing_period_mul == 0 and self._steps >= self._cosine_annealing_period + self._warmup_iters:
            lr = self._min_lr
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        elif self._steps <= self._warmup_iters:
            # Warmup: adjust learning rate linearly
            # (max_lr - initial_lr) / warmup_iters
            lr = self._initial_lr + self._steps * self._lr_increment
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if not self._cosine_annealing_base_lr:
                self._cosine_annealing_base_lr = self.cur_lr

            """每步更新学习率"""
            # 计算当前周期的最大步数
            T_max = self._cosine_annealing_period * (max(self._cosine_annealing_period_mul, 1) ** self.cycle)

            # 更新周期状态
            self.T_cur += 1
            if self.T_cur >= T_max:
                self.cycle += 1
                self.T_cur = 0  # 重置周期步数

            # 计算并设置新学习率
            cos_factor = (1 + math.cos(math.pi * self.T_cur / T_max)) / 2
            lr = self._min_lr + (self._cosine_annealing_base_lr - self._min_lr) * cos_factor

            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr

        self._current_lr = lr

        if self.need_log:
            log(f"step={self.cur_steps},lr={lr}\n", f'{get_log_dir()}lr.txt')

    def get_ckpt_dict(self) -> dict:
        return {
            'cur_lr': self._current_lr,
            'lr_steps': self.cur_steps,
            'cosine_annealing_base_lr': self._cosine_annealing_base_lr,
            't_cur': self.T_cur,
            'cycle': self.cycle,
        }

    def restore_ckpt_dict(self, ckpt: dict):
        if ckpt['cur_lr']:
            self._current_lr = ckpt['cur_lr']

        if ckpt['lr_steps']:
            self._steps = ckpt['lr_steps']

        if ckpt['cosine_annealing_base_lr']:
            self._cosine_annealing_base_lr = ckpt['cosine_annealing_base_lr']

        if ckpt['t_cur']:
            self.T_cur = ckpt['t_cur']

        if ckpt['cycle']:
            self.cycle = ckpt['cycle']

        self._update_lr()


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

    def can_clip_grad(self):
        return True

    def get_ckpt_dict(self) -> dict:
        return {'cur_lr': self._current_lr}

    def restore_ckpt_dict(self, ckpt: dict):
        if ckpt['cur_lr']:
            self._current_lr = ckpt['cur_lr']