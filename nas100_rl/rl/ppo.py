"""Compact PPO with GAE and a KL anchor to the "always take 1.0x" prior.

Anti-overfitting devices baked in:
- tiny networks (2x64) with LayerNorm + dropout + weight decay;
- the policy head is initialized AT the prior (known-profitable always-take), and a
  KL(pi || prior) penalty makes deviations cost evidence;
- entropy floor; feature dropout during training (robustness jitter);
- multi-seed training with a deterministic mean-logits ensemble at deployment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 3
PRIOR_PROBS = np.array([0.06, 0.10, 0.84], dtype=np.float64)  # skip, 0.5x, 1.0x


def prior_logits() -> torch.Tensor:
    return torch.log(torch.tensor(PRIOR_PROBS, dtype=torch.float32))


def make_prior_table(best_action: dict[int, int], mass: float = 0.65) -> torch.Tensor:
    """Per-strategy anchor priors: `mass` on the strategy's train-optimal static
    multiplier, the rest spread over the other actions (soft anchor: leaves room
    for conditional deviations)."""
    tbl = torch.full((3, N_ACTIONS), (1.0 - mass) / (N_ACTIONS - 1), dtype=torch.float32)
    for k, a in best_action.items():
        tbl[k, a] = mass
    return torch.log(tbl)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.head = nn.Linear(hidden, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        with torch.no_grad():
            self.head.bias.copy_(prior_logits())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.97
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 4
    minibatch: int = 512
    ent_coef: float = 0.005
    kl_coef: float = 0.5          # anchor strength to the always-take prior
    vf_coef: float = 0.5
    grad_clip: float = 0.5
    feat_dropout: float = 0.05    # input-feature dropout during collection (jitter)
    feat_noise: float = 0.1       # additive Gaussian noise on z-scored features,
                                  # TRAINING COLLECTION ONLY (never at evaluation)


class PPO:
    def __init__(self, obs_dim: int, cfg: PPOConfig, seed: int, device: str = "cpu",
                 prior_table: torch.Tensor | None = None):
        self.cfg = cfg
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31))
        self.rng = np.random.default_rng(seed)
        self.pi = PolicyNet(obs_dim).to(device)
        self.vf = ValueNet(obs_dim).to(device)
        self.opt = torch.optim.AdamW(
            list(self.pi.parameters()) + list(self.vf.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
        # per-strategy anchor priors (rows: strategy idx); fallback: uniform prior row
        if prior_table is None:
            self.prior_table = prior_logits().unsqueeze(0).repeat(3, 1).to(device)
        else:
            self.prior_table = prior_table.to(device)
        with torch.no_grad():
            self.pi.head.bias.copy_(self.prior_table.log_softmax(-1).exp().mean(0).log())
        self.update_count = 0

    # -------------------------------------------------------------- collection
    @torch.no_grad()
    def act_batch(self, obs: np.ndarray, stochastic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if stochastic and self.cfg.feat_dropout > 0:
            mask = torch.rand_like(x) > self.cfg.feat_dropout
            x = x * mask
        if stochastic and self.cfg.feat_noise > 0:
            x = x + self.cfg.feat_noise * torch.randn_like(x)
        self.pi.train(stochastic)  # dropout active during collection only
        logits = self.pi(x)
        self.pi.train(False)
        v = self.vf(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample() if stochastic else logits.argmax(-1)
        logp = dist.log_prob(a)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def collect(self, make_env, windows: list[tuple[int, int]],
                strat_idx: np.ndarray | None = None) -> dict:
        """Run len(windows) episodes in lockstep with batched forwards.
        strat_idx: per-signal strategy index array (for per-sample anchor priors)."""
        envs = [make_env() for _ in windows]
        obs = [e.reset(i0, i1) for e, (i0, i1) in zip(envs, windows)]
        alive = list(range(len(envs)))
        store = {k: [[] for _ in envs] for k in ("obs", "act", "logp", "val", "rew", "sidx")}
        while alive:
            ob = np.stack([obs[j] for j in alive])
            a, lp, v = self.act_batch(ob, stochastic=True)
            nxt_alive = []
            for n, j in enumerate(alive):
                store["obs"][j].append(obs[j])
                store["act"][j].append(a[n])
                store["logp"][j].append(lp[n])
                store["val"][j].append(v[n])
                store["sidx"][j].append(0 if strat_idx is None else int(strat_idx[envs[j].i]))
                o2, r, done, _ = envs[j].step(int(a[n]))
                store["rew"][j].append(r)
                if not done:
                    obs[j] = o2
                    nxt_alive.append(j)
            alive = nxt_alive
        return store

    # -------------------------------------------------------------- update
    def update(self, store: dict) -> dict:
        cfg = self.cfg
        obs_l, act_l, logp_l, adv_l, ret_l = [], [], [], [], []
        for j in range(len(store["obs"])):
            rew = np.asarray(store["rew"][j], dtype=np.float64)
            val = np.asarray(store["val"][j], dtype=np.float64)
            T = len(rew)
            adv = np.zeros(T)
            last = 0.0
            for t in reversed(range(T)):
                nv = val[t + 1] if t + 1 < T else 0.0
                delta = rew[t] + cfg.gamma * nv - val[t]
                last = delta + cfg.gamma * cfg.lam * last
                adv[t] = last
            ret = adv + val
            obs_l.append(np.stack(store["obs"][j]))
            act_l.append(np.asarray(store["act"][j]))
            logp_l.append(np.asarray(store["logp"][j]))
            adv_l.append(adv)
            ret_l.append(ret)
        obs = torch.as_tensor(np.concatenate(obs_l), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.concatenate(act_l), dtype=torch.long, device=self.device)
        sidx = torch.as_tensor(np.concatenate([np.asarray(s) for s in store["sidx"]]),
                               dtype=torch.long, device=self.device)
        logp_old = torch.as_tensor(np.concatenate(logp_l), dtype=torch.float32, device=self.device)
        adv = np.concatenate(adv_l)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(np.concatenate(ret_l), dtype=torch.float32, device=self.device)

        n = len(obs)
        idx = np.arange(n)
        stats = dict(pi_loss=0.0, v_loss=0.0, kl_prior=0.0, entropy=0.0, n=n)
        for _ in range(cfg.epochs):
            self.rng.shuffle(idx)
            for k in range(0, n, cfg.minibatch):
                mb = torch.as_tensor(idx[k:k + cfg.minibatch], device=self.device)
                logits = self.pi(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                pi_loss = -torch.min(s1, s2).mean()
                ent = dist.entropy().mean()
                logq = torch.log_softmax(logits, dim=-1)
                prior_mb = self.prior_table.log_softmax(dim=-1)[sidx[mb]]
                kl_prior = (logq.exp() * (logq - prior_mb)).sum(-1).mean()
                v = self.vf(obs[mb])
                v_loss = ((v - ret[mb]) ** 2).mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent \
                    + cfg.kl_coef * kl_prior
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.pi.parameters()) + list(self.vf.parameters()), cfg.grad_clip)
                self.opt.step()
                stats["pi_loss"] += float(pi_loss.detach())
                stats["v_loss"] += float(v_loss.detach())
                stats["kl_prior"] += float(kl_prior.detach())
                stats["entropy"] += float(ent.detach())
        nb = cfg.epochs * math.ceil(n / cfg.minibatch)
        for k in ("pi_loss", "v_loss", "kl_prior", "entropy"):
            stats[k] /= nb
        self.update_count += 1
        return stats

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return dict(pi=self.pi.state_dict(), vf=self.vf.state_dict(),
                    opt=self.opt.state_dict(), update_count=self.update_count,
                    cfg=self.cfg.__dict__)

    def load_state_dict(self, sd: dict) -> None:
        self.pi.load_state_dict(sd["pi"])
        self.vf.load_state_dict(sd["vf"])
        self.opt.load_state_dict(sd["opt"])
        self.update_count = sd.get("update_count", 0)


class EnsemblePolicy:
    """Deterministic deployment policy: mean logits across seed models, argmax."""

    def __init__(self, models: list[PolicyNet], device: str = "cpu"):
        self.models = models
        self.device = device
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, i: int = -1) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = torch.stack([m(x) for m in self.models]).mean(0)
        return int(logits.argmax(-1).item())

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = torch.stack([m(x) for m in self.models]).mean(0)
        return torch.softmax(logits, -1).squeeze(0).cpu().numpy()
