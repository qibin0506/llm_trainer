from typing import List, Dict, Any, Optional
import math

import torch

from .utils import is_bf16_supported


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1.0, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0

    return lr * adjusted_ratio


def _zeropower_via_newtonschulz5(
        G: torch.Tensor,
        steps: int = 5,
        eps: float = 1e-7
) -> torch.Tensor:
    """通过 5 阶 Newton-Schulz 迭代计算正交矩阵（极分解）"""
    assert len(G.shape) == 2, "Muon only supports 2D Matrix"

    a, b, c = (3.4445, -4.7750, 2.0315)

    if G.dtype == torch.bfloat16:
        X = G.clone()
    elif is_bf16_supported(G):
        X = G.bfloat16()
    else:
        X = G.float()

    X = X / (X.norm().clamp(min=eps))

    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X.to(dtype=G.dtype)


class MuonOptim(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 0.02,
            weight_decay: float = 0.01,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            adjust_lr_fn: Optional[str] = None
    ):
        if hasattr(torch, "compile"):
            try:
                self.zeropower_via_newtonschulz5 = torch.compile(_zeropower_via_newtonschulz5)
            except:
                self.zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5
        else:
            self.zeropower_via_newtonschulz5 = _zeropower_via_newtonschulz5

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adjust_lr_fn=adjust_lr_fn
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            adjust_lr_fn = group.get("adjust_lr_fn", None)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse grad")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                effective_lr = lr
                if update.ndim == 2:
                    update = self.zeropower_via_newtonschulz5(update, steps=ns_steps)
                    effective_lr = _adjust_lr(lr, adjust_lr_fn, update.shape)

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(update, alpha=-effective_lr)

        return loss


try:
    Muon = torch.optim.Muon
except (ImportError, AttributeError):
    Muon = MuonOptim


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(
            self,
            muon_params: List[Dict[str, Any]],
            adamw_params: List[Dict[str, Any]],
            muon_lr: float = 0.02,
            muon_momentum: float = 0.95,
            muon_weight_decay: float = 0.01,
            muon_ns_steps: int = 5,
            muon_adjust_lr_fn: Optional[str] = None,
            adamw_lr: float = 1e-3,
            adamw_betas: tuple = (0.9, 0.95),
            adamw_weight_decay: float = 0.01,
            adamw_cls=torch.optim.AdamW
    ):
        muon_params = [g for g in muon_params if len(g.get('params', [])) > 0]
        adamw_params = [g for g in adamw_params if len(g.get('params', [])) > 0]
        if not muon_params and not adamw_params:
            raise ValueError("Optimizer got an empty parameter list")

        self.muon_optim = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn
        )
        self.adamw_optim = adamw_cls(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            weight_decay=adamw_weight_decay
        )

        param_groups = self.muon_optim.param_groups + self.adamw_optim.param_groups
        defaults = {**self.muon_optim.defaults, **self.adamw_optim.defaults}
        super().__init__(param_groups, defaults)

        self.muon_optim.state = self.state
        self.adamw_optim.state = self.state

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_optim.step()
        self.adamw_optim.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.muon_optim.zero_grad(set_to_none=set_to_none)
        self.adamw_optim.zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)

        self.muon_optim.state = self.state
        self.adamw_optim.state = self.state

        num_muon_groups = len(self.muon_optim.param_groups)
        self.muon_optim.param_groups = self.param_groups[:num_muon_groups]
        self.adamw_optim.param_groups = self.param_groups[num_muon_groups:]


def _get_ds_muon(params):
    try:
        from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
        valid_groups = [g for g in params if len(g.get("params", [])) > 0]
        if not valid_groups:
            return None

        for group in valid_groups:
            is_muon = group.get("use_muon", False)
            for p in group.get("params", []):
                p.use_muon = is_muon

        return MuonWithAuxAdam(valid_groups)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None
