import torch
import math
from .utils import log

class LRScheduler:
    @property
    def cur_steps(self):
        raise NotImplementedError()

    def update_steps(self, steps):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def can_clip_grad(self):
        raise NotImplementedError()



class CosineAnnealingWarmupLRScheduler(LRScheduler):
    def __init__(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            warmup_iters,
            initial_lr,
            min_lr,
            max_lr,
            total_iters,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iters = total_iters
        self.lr_increment = (max_lr - initial_lr) / warmup_iters
        self.steps = -1

        log(f'warmup_iters: {self.warmup_iters},'
              f' initial_lr: {self.initial_lr},'
              f' min_lr: {self.min_lr},'
              f' max_lr: {self.max_lr},'
              f'total_iters: {self.total_iters},'
              f'lr_increment: {self.lr_increment}')

    @property
    def cur_steps(self):
        return self.steps

    def update_steps(self, steps):
        log(f'update step to {steps}')
        self.steps = steps
        self._update_lr()

    def step(self):
        self.steps += 1
        self._update_lr()

    def can_clip_grad(self):
        return self.steps > self.warmup_iters

    def _update_lr(self):
        if self.steps <= self.warmup_iters:
            # Warmup: adjust learning rate linearly
            lr = self.initial_lr + self.steps * self.lr_increment
        else:
            # Cosine annealing phase
            progress = (self.steps - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class NoneLRScheduler(LRScheduler):
    @property
    def cur_steps(self):
        return -1

    def update_steps(self, steps):
        pass

    def step(self):
        pass

    def can_clip_grad(self):
        return True
