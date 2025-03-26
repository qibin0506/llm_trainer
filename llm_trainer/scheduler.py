from abc import ABC, abstractmethod
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .log import (
    log,
    get_log_dir
)

class LRScheduler(ABC):
    @property
    @abstractmethod
    def cur_steps(self):
        pass

    @property
    @abstractmethod
    def cur_lr(self):
        pass

    @abstractmethod
    def update_steps(self, steps):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def can_clip_grad(self):
        pass


class WarmupCosineAnnealingLRScheduler(LRScheduler):
    def __init__(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            initial_lr: float,
            min_lr: float,
            max_lr: float,
            warmup_iters: int,
            period: int, # 每个周期的步数
            period_mul: int = 1, # 周期长度的倍数
            need_log: bool = False
    ):
        super().__init__()

        self._optimizer = optimizer
        self._initial_lr = initial_lr
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._warmup_iters = warmup_iters
        self._period = period
        self._period_mul = period_mul

        self._lr_increment = (max_lr - initial_lr) / warmup_iters
        self._steps = -1
        self._current_lr = initial_lr

        self._cosine_annealing_scheduler = None

        self.need_log = need_log


    @property
    def cur_steps(self):
        return self._steps

    @property
    def cur_lr(self):
        return self._current_lr

    def update_steps(self, steps):
        log(f'update step to {steps}')
        self._steps = steps
        self._update_lr()

    def step(self):
        self._steps += 1
        self._update_lr()

    def can_clip_grad(self):
        return self._steps > self._warmup_iters

    def _update_lr(self):
        if self._steps <= self._warmup_iters:
            # Warmup: adjust learning rate linearly
            lr = self._initial_lr + self._steps * self._lr_increment
        else:
            if not self._cosine_annealing_scheduler:
                self._cosine_annealing_scheduler = CosineAnnealingWarmRestarts(
                    optimizer=self._optimizer,
                    T_0=self._period,
                    T_mult=self._period_mul,
                    eta_min=self._min_lr
                )

            self._cosine_annealing_scheduler.step()
            lr = self._cosine_annealing_scheduler.get_last_lr()[0]

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

        self._current_lr = lr

        if self.need_log:
            log(f"step={self.cur_steps},lr={self.cur_lr}\n", f'{get_log_dir()}lr.txt')


class NoneLRScheduler(LRScheduler):
    def __init__(self, initial_lr):
        self._current_lr = initial_lr

    @property
    def cur_steps(self):
        return -1

    @property
    def cur_lr(self):
        return self._current_lr

    def update_steps(self, steps):
        pass

    def step(self):
        pass

    def can_clip_grad(self):
        return True
