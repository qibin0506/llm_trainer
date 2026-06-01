import torch
from .train_configs import OptimConfig


def get_muon_args(config: OptimConfig):
    muon_momentum = config.muon_momentum
    if muon_momentum is None:
        muon_momentum = config.betas[0] if config.betas else 0.95
    muon_nesterov = config.muon_nesterov
    muon_ns_steps = config.muon_ns_steps

    return muon_momentum, muon_nesterov, muon_ns_steps


class MuonAdamW(torch.optim.Optimizer):
    def __init__(
            self,
            m_params,
            a_decay_params,
            a_no_decay_params,
            lr,
            muon_lr_multiplier,
            betas,
            weight_decay,
            m_momentum,
            m_nesterov,
            m_ns_steps
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        all_params = m_params + a_decay_params + a_no_decay_params
        super().__init__(all_params, defaults)

        self.muon = torch.optim.Muon(
            m_params,
            lr=lr * muon_lr_multiplier,
            weight_decay=weight_decay,
            momentum=m_momentum,
            nesterov=m_nesterov,
            ns_steps=m_ns_steps
        ) if m_params else None

        self.adamw = torch.optim.AdamW([
            {"params": a_decay_params, "weight_decay": weight_decay},
            {"params": a_no_decay_params, "weight_decay": 0.0},
        ], lr=lr, betas=betas)

        self.param_groups = []
        if self.muon:
            for g in self.muon.param_groups:
                g['muon_multiplier'] = muon_lr_multiplier
            self.param_groups += self.muon.param_groups

        self.param_groups += self.adamw.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon:
            self.muon.step()
        self.adamw.step()

        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon: self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict() if self.muon else {}, "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        if self.muon and "muon" in state_dict: self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


class PPOMuonAdamW(torch.optim.Optimizer):
    def __init__(
            self,
            policy_config: OptimConfig,
            value_config: OptimConfig,
            p_muon,
            v_muon,
            adamw_groups,
            p_weight_decay,
            v_weight_decay,
            p_momentum,
            p_nesterov,
            p_ns_steps,
            v_momentum,
            v_nesterov,
            v_ns_steps
    ):
        defaults = dict(lr=policy_config.initial_lr)
        all_params = p_muon + v_muon + [p for g in adamw_groups for p in g["params"]]
        super().__init__(all_params, defaults)

        self.muon_policy = torch.optim.Muon(
            p_muon,
            lr=policy_config.initial_lr * policy_config.muon_lr_multiplier,
            weight_decay=p_weight_decay,
            momentum=p_momentum,
            nesterov=p_nesterov,
            ns_steps=p_ns_steps
        ) if p_muon else None

        self.muon_value = torch.optim.Muon(
            v_muon,
            lr=value_config.initial_lr * value_config.muon_lr_multiplier,
            weight_decay=v_weight_decay,
            momentum=v_momentum,
            nesterov=v_nesterov,
            ns_steps=v_ns_steps
        ) if v_muon else None

        self.adamw = torch.optim.AdamW(adamw_groups)

        self.param_groups = []
        if self.muon_policy:
            for g in self.muon_policy.param_groups:
                g['name'] = 'policy_muon'
                g['muon_multiplier'] = policy_config.muon_lr_multiplier
            self.param_groups += self.muon_policy.param_groups

        if self.muon_value:
            for g in self.muon_value.param_groups:
                g['name'] = 'value_muon'
                g['muon_multiplier'] = value_config.muon_lr_multiplier
            self.param_groups += self.muon_value.param_groups

        self.param_groups += self.adamw.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.muon_policy:
            self.muon_policy.step()
        if self.muon_value:
            self.muon_value.step()
        self.adamw.step()

        return loss

    def zero_grad(self, set_to_none=True):
        if self.muon_policy: self.muon_policy.zero_grad(set_to_none)
        if self.muon_value: self.muon_value.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self):
        return {
            "muon_policy": self.muon_policy.state_dict() if self.muon_policy else {},
            "muon_value": self.muon_value.state_dict() if self.muon_value else {},
            "adamw": self.adamw.state_dict()
        }

    def load_state_dict(self, state_dict):
        if self.muon_policy and "muon_policy" in state_dict: self.muon_policy.load_state_dict(state_dict["muon_policy"])
        if self.muon_value and "muon_value" in state_dict: self.muon_value.load_state_dict(state_dict["muon_value"])
        self.adamw.load_state_dict(state_dict["adamw"])