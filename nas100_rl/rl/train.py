"""Training orchestration: purged walk-forward grid -> multi-seed final training.

Modes:
  grid  : small hyperparameter grid scored on 3 expanding walk-forward folds inside
          TRAIN (5-day embargo). Picks (config, update budget) by mean fold score.
  final : trains N seeds on the full TRAIN range with the chosen config/budget,
          checkpointing every `snap_every` updates (resumable).

Fold score = fold Sharpe + min(net_ratio_vs_always, 1.5)  (Sharpe is the binding
half of the success gate; the net term keeps profit from collapsing).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import config
from . import evaluate as ev
from .env import SignalEnv, native_qty
from .ppo import PPO, PPOConfig, EnsemblePolicy, make_prior_table

CKPT = config.CHECKPOINT_DIR
EMBARGO_MS = 5 * 86_400_000
WINDOW_MS = 90 * 86_400_000
STRAT_IDX = {"s2": 0, "dmi": 1, "cpmt": 2}


def strat_idx_array(sig) -> np.ndarray:
    return sig["strategy"].map(STRAT_IDX).to_numpy()


def static_prior_table(world: dict, lo: int, hi: int, kappa: float):
    """Train-only static mean-variance optimum per strategy -> anchor prior table.
    Uses the SAME utility as the env reward; data restricted to signals [lo, hi)."""
    sig = world["sig"].iloc[lo:hi]
    e0 = config.RL["initial_capital"]
    best = {}
    for name, k in STRAT_IDX.items():
        s = sig[sig["strategy"] == name]
        if len(s) < 30:
            best[k] = 2
            continue
        rs = 100.0 * np.array([native_qty(name, e0, sd) for sd in s["stop_dist"]]) \
            * s["pc_pnl"].to_numpy() / e0
        utils = [float(np.mean(w * rs - 0.5 * kappa * (w * rs) ** 2))
                 for w in config.RL["action_multipliers"]]
        best[k] = int(np.argmax(utils))
    return make_prior_table(best), best


def sample_windows(rng: np.random.Generator, sig_ms: np.ndarray, lo: int, hi: int,
                   n: int, min_steps: int = 20) -> list[tuple[int, int]]:
    """Random ~90-trading-day episode windows fully inside [lo, hi] (signal indices)."""
    out = []
    ms_lo = sig_ms[lo]
    ms_hi = sig_ms[hi - 1]
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        # 90 trading days ~ 126 calendar days
        span = int(WINDOW_MS * 1.4)
        if ms_hi - span <= ms_lo:
            start_ms = ms_lo
        else:
            start_ms = int(rng.integers(ms_lo, ms_hi - span))
        i0 = int(np.searchsorted(sig_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(sig_ms, start_ms + span, side="right"))
        i0 = max(i0, lo)
        i1 = min(i1, hi)
        if i1 - i0 >= min_steps:
            out.append((i0, i1))
    if not out:
        out = [(lo, hi)] * n
    return out


def train_run(world: dict, ppo_cfg: PPOConfig, env_kwargs: dict, seed: int,
              lo: int, hi: int, updates: int, episodes_per_update: int = 16,
              snap_every: int = 10, ckpt_prefix: str | None = None,
              eval_fold: tuple[int, int] | None = None,
              resume: bool = False) -> dict:
    sig_ms = world["sig"]["signal_ms"].to_numpy()
    obs_dim = world["feats"].shape[1] + 4
    # anchor at native 1.0x for every strategy: blanket re-leverage (up or down)
    # proved Sharpe-invariant; only conditional selection moves the binding metric.
    # The variance-penalized reward still teaches conditional 0.5x/skip.
    prior_actions = {0: 2, 1: 2, 2: 2}
    prior_table = make_prior_table(prior_actions, mass=0.84)
    sidx = strat_idx_array(world["sig"])
    agent = PPO(obs_dim, ppo_cfg, seed=seed, prior_table=prior_table)
    rng = np.random.default_rng(seed * 7919 + 13)
    start_u = 0
    if resume and ckpt_prefix:
        last = sorted(CKPT.glob(f"{ckpt_prefix}_u*.pt"),
                      key=lambda p: int(p.stem.split("_u")[-1]))
        if last:
            sd = torch.load(last[-1], map_location="cpu", weights_only=False)
            agent.load_state_dict(sd["agent"])
            rng = np.random.default_rng()
            rng.bit_generator.state = sd["rng_state"]
            start_u = sd["update"]
            print(f"  resumed {ckpt_prefix} at update {start_u}")

    def make_env():
        return SignalEnv(world["sig"], world["feats"], world["mark"], **env_kwargs)

    history = []
    for u in range(start_u, updates):
        wins = sample_windows(rng, sig_ms, lo, hi, episodes_per_update)
        store = agent.collect(make_env, wins, strat_idx=sidx)
        stats = agent.update(store)
        rec = dict(update=u + 1, **{k: round(v, 4) for k, v in stats.items() if k != "n"})
        if eval_fold is not None and (u + 1) % snap_every == 0:
            pol = EnsemblePolicy([agent.pi])
            m = ev.eval_range(world, eval_fold[0], eval_fold[1], pol)
            rec["fold_sharpe"] = round(m["sharpe"], 3)
            rec["fold_net"] = round(m["net"], 0)
        history.append(rec)
        if ckpt_prefix and ((u + 1) % snap_every == 0 or u + 1 == updates):
            CKPT.mkdir(exist_ok=True)
            torch.save(dict(agent=agent.state_dict(), update=u + 1,
                            rng_state=rng.bit_generator.state,
                            env_kwargs=env_kwargs, seed=seed,
                            prior_actions=prior_actions),
                       CKPT / f"{ckpt_prefix}_u{u + 1}.pt")
    return dict(agent=agent, history=history, prior_actions=prior_actions)


def grid_mode(seeds: int = 2, updates: int = 60, basis: str = "tv") -> dict:
    world = ev.load_all(basis=basis)
    sig_ms = world["sig"]["signal_ms"].to_numpy()
    t0, t1 = world["periods"]["train"]
    span = sig_ms[t1 - 1] - sig_ms[t0]
    # six ~5-month eval folds: the success gate compares against the best solo over
    # SHORT windows (val 7mo, oos 5mo) where the max-of-3-solos benchmark is luck-
    # inflated; folds must mimic that geometry
    folds = []
    for k in range(6):
        fa, fb = 0.4 + 0.1 * k, 0.5 + 0.1 * k
        f0 = int(np.searchsorted(sig_ms, sig_ms[t0] + fa * span))
        f1 = int(np.searchsorted(sig_ms, sig_ms[t0] + fb * span, side="right"))
        f1 = min(f1, t1)
        train_hi = int(np.searchsorted(sig_ms, sig_ms[f0] - EMBARGO_MS))
        folds.append((t0, train_hi, f0, f1))

    # per-fold gate references: best-solo net (constraint) and best-solo Sharpe
    # (the margin target)
    strat = world["sig"]["strategy"].to_numpy()
    solo_net = {}
    solo_sharpe = {}
    for k, (a, b, f0, f1) in enumerate(folds):
        nets, shs = [], []
        for name in ("s2", "dmi", "cpmt"):
            m = ev.eval_range(world, f0, f1,
                              lambda obs, i, n=name: 2 if strat[i] == n else 0)
            nets.append(m["net"])
            shs.append(m["sharpe"])
        solo_net[k] = max(nets)
        solo_sharpe[k] = max(shs)
        m = ev.eval_range(world, f0, f1, lambda obs, i: 2)
        print(f"fold{k}: eval [{f0}:{f1})  always sharpe={m['sharpe']:.2f} "
              f"net={m['net']:,.0f}  best-solo sharpe={solo_sharpe[k]:.2f} "
              f"net={solo_net[k]:,.0f}")

    def fold_score(h, k):
        # gate margin: Sharpe above the fold's best solo, net as hard constraint
        return (h["fold_sharpe"] - solo_sharpe[k]) \
            - (2.0 if h["fold_net"] <= solo_net[k] else 0.0)

    grid = []
    for kappa in (0.03, 0.08):
        for kl in (0.05, 0.15):
            grid.append(dict(var_penalty=kappa, kl_coef=kl))

    results = []
    for gi, g in enumerate(grid):
        scores = []
        budgets = []
        for k, (a, b, f0, f1) in enumerate(folds):
            for s in range(seeds):
                # contextual-bandit credit: immediate per-trade reward, gamma=0
                cfg = PPOConfig(kl_coef=g["kl_coef"], gamma=0.0, lam=0.0)
                envk = dict(var_penalty=g["var_penalty"], credit="trade")
                r = train_run(world, cfg, envk, seed=1000 * gi + 10 * k + s,
                              lo=a, hi=b, updates=updates, eval_fold=(f0, f1),
                              snap_every=5)
                hh = [h for h in r["history"] if "fold_sharpe" in h]
                # stability selection: score each snapshot as the mean of itself and
                # its two predecessors (penalizes one-off lucky snapshots)
                sc = [fold_score(h, k) for h in hh]
                smooth = [float(np.mean(sc[max(0, j - 2):j + 1])) for j in range(len(sc))]
                j_best = int(np.argmax(smooth))
                scores.append(smooth[j_best])
                budgets.append(hh[j_best]["update"])
        results.append(dict(**g, score=float(np.mean(scores)),
                            budget=int(np.median(budgets))))
        print(f"grid {g} -> score {results[-1]['score']:.3f} budget {results[-1]['budget']}")
    best = max(results, key=lambda r: r["score"])
    suffix = "" if basis == "tv" else f"_{basis}"
    (CKPT / f"grid_result{suffix}.json").write_text(json.dumps(dict(best=best, all=results), indent=2))
    print("best config:", best)
    return best


def final_mode(seeds: int = 5, resume: bool = False, fit_through: str = "train",
               basis: str = "tv") -> None:
    """fit_through="train": candidate for the val gate. fit_through="val": same
    recipe refit on train+val (expanding-window deployment) for the OOS test —
    only run after the recipe passed the val gate."""
    world = ev.load_all(basis=basis)
    suffix = "" if basis == "tv" else f"_{basis}"
    best = json.loads((CKPT / f"grid_result{suffix}.json").read_text())["best"]
    t0, t1 = world["periods"]["train"]
    if fit_through == "val":
        t1 = world["periods"]["val"][1]
    updates = int(best["budget"])  # walk-forward-validated early stop
    tag = ("final" if fit_through == "train" else "deploy") + suffix
    print(f"{tag} training (fit through {fit_through}): config={best} "
          f"updates={updates} seeds={seeds}")
    manifest = dict(config=best, updates=updates, seeds=[], fit_through=fit_through)
    for s in range(seeds):
        t_start = time.time()
        cfg = PPOConfig(kl_coef=best["kl_coef"], gamma=0.0, lam=0.0)
        envk = dict(var_penalty=best["var_penalty"], credit="trade")
        r = train_run(world, cfg, envk, seed=42 + s, lo=t0, hi=t1, updates=updates,
                      ckpt_prefix=f"{tag}_seed{s}", snap_every=10, resume=resume)
        torch.save(dict(agent=r["agent"].state_dict(), env_kwargs=envk, seed=42 + s),
                   CKPT / f"{tag}_seed{s}.pt")
        manifest["seeds"].append(f"{tag}_seed{s}.pt")
        manifest["prior_actions"] = r["prior_actions"]
        print(f"seed {s}: {time.time() - t_start:.0f}s "
              f"(last kl={r['history'][-1]['kl_prior']:.3f} ent={r['history'][-1]['entropy']:.3f})")
    (CKPT / f"ensemble_manifest_{tag}.json").write_text(json.dumps(manifest, indent=2))


def load_ensemble(tag: str = "final") -> EnsemblePolicy:
    from .ppo import PolicyNet
    manifest = json.loads((CKPT / f"ensemble_manifest_{tag}.json").read_text())
    models = []
    for f in manifest["seeds"]:
        sd = torch.load(CKPT / f, map_location="cpu", weights_only=False)
        world_dim = sd["agent"]["pi"]["trunk.0.weight"].shape[1]
        m = PolicyNet(world_dim)
        m.load_state_dict(sd["agent"]["pi"])
        models.append(m)
    return EnsemblePolicy(models)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "final"], required=True)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fit-through", choices=["train", "val"], default="train")
    ap.add_argument("--basis", choices=["tv", "live"], default="tv")
    args = ap.parse_args()
    if args.mode == "grid":
        grid_mode(seeds=args.seeds or 2, updates=args.updates, basis=args.basis)
    else:
        final_mode(seeds=args.seeds or 5, resume=args.resume, fit_through=args.fit_through,
                   basis=args.basis)
