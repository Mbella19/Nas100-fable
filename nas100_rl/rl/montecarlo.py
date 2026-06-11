"""Monte Carlo overfitting assessment of the deployed RL ensemble.

All tests run on HELD-OUT data (validation + locked OOS, 2025-06..2026-06), where
overfitting actually manifests; the train-period null is computed only to measure
the in-sample vs held-out generalization gap.

A) Paired stationary block bootstrap (Politis-Romano) of daily returns:
   resample the SAME day blocks for RL, always-take, and the three solos ->
   distribution of Sharpe/net/MaxDD and of the (RL - best solo) Sharpe difference.
   Pairing removes market-sample luck from the comparison.

B) Action-matched random-policy null: random policies that reproduce the RL's
   per-strategy action frequencies (skip/half/full) but choose WHICH signals at
   random -> where does the real policy sit in the null distribution? If the
   percentile is high on train but ~50% held-out, the selection is overfit.

C) Deflated Sharpe Ratio (Bailey & Lopez de Prado) of the held-out year, with
   trials = number of logged OOS candidate evaluations, non-normality corrected.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sstats

from .. import config
from . import evaluate as ev
from .env import SignalEnv
from .ppo import EnsemblePolicy


# ---------------------------------------------------------------- helpers
def daily_returns(world: dict, i0: int, i1: int, policy) -> tuple[np.ndarray, pd.DataFrame]:
    env = SignalEnv(world["sig"], world["feats"], world["mark"])
    from .env import run_fixed_policy
    res = run_fixed_policy(env, i0, i1, policy)
    ms_lo = int(world["sig"]["signal_ms"].iloc[i0])
    ms_hi = int(world["sig"]["signal_ms"].iloc[i1 - 1])
    curve = ev.daily_curve_range(res["ledger"], world["sig"], world["days"], ms_lo, ms_hi, env.e0)
    eq = curve["equity"].to_numpy()
    return np.diff(np.log(np.maximum(eq, 1e-9))), curve


def sharpe(r: np.ndarray) -> float:
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0


def maxdd(r: np.ndarray) -> float:
    eq = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(eq)
    return float(((peak - eq) / peak).max())


def stationary_bootstrap_idx(n: int, n_draws: int, mean_block: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_draws, n), wrap-around."""
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_draws, n))
    cont = rng.random(size=(n_draws, n)) >= p     # continue previous block
    idx = np.empty((n_draws, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % n, starts[:, t])
    return idx


def action_frequencies(world: dict, policy, i0: int, i1: int) -> dict[str, np.ndarray]:
    env = SignalEnv(world["sig"], world["feats"], world["mark"])
    obs = env.reset(i0, i1)
    strat = world["sig"]["strategy"].to_numpy()
    acts: dict[str, list[int]] = {"s2": [], "dmi": [], "cpmt": []}
    done = False
    while not done:
        a = policy(obs, env.i)
        acts[strat[env.i]].append(a)
        obs, _, done, _ = env.step(a)
    n_actions = len(config.RL["action_multipliers"])
    freq = {}
    for s, lst in acts.items():
        c = np.bincount(lst, minlength=n_actions).astype(float)
        freq[s] = c / max(1, c.sum())
    return freq


def matched_random_policy(strat: np.ndarray, freq: dict[str, np.ndarray], seed: int):
    rng = np.random.default_rng(seed)
    n_actions = len(config.RL["action_multipliers"])

    def pol(obs, i):
        return int(rng.choice(n_actions, p=freq[strat[i]]))
    return pol


def deflated_sharpe(r: np.ndarray, sr_benchmark_annual: float, n_trials: int) -> dict:
    """PSR/DSR (Bailey & Lopez de Prado). Daily series; SR0 from the benchmark and
    the expected max of n_trials random draws."""
    n = len(r)
    sr_d = r.mean() / r.std()                      # daily Sharpe
    g3 = float(sstats.skew(r))
    g4 = float(sstats.kurtosis(r, fisher=False))
    # expected max daily-SR of n_trials trials around the benchmark (Euler-Mascheroni)
    var_sr = (1.0 / n) * (1 - g3 * sr_d + (g4 - 1) / 4 * sr_d ** 2)
    var_sr = max(var_sr, 1e-12)
    em = 0.5772156649
    if n_trials > 1:
        z1 = sstats.norm.ppf(1 - 1.0 / n_trials)
        z2 = sstats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0_d = sr_benchmark_annual / np.sqrt(252) + np.sqrt(var_sr) * ((1 - em) * z1 + em * z2)
    else:
        sr0_d = sr_benchmark_annual / np.sqrt(252)
    psr = sstats.norm.cdf((sr_d - sr0_d) / np.sqrt(var_sr))
    return dict(sr_annual=sr_d * np.sqrt(252), sr0_annual=sr0_d * np.sqrt(252),
                dsr_prob=float(psr), skew=g3, kurt=g4, n_days=n)


# ---------------------------------------------------------------- main
def run(n_boot: int = 10_000, n_null_held: int = 2_000, n_null_train: int = 1_000,
        mean_block: float = 10.0, seed: int = 0, basis: str = "tv",
        ensemble_tag: str = "deploy") -> dict:
    world = ev.load_all(basis=basis)
    from .train import load_ensemble
    pol = load_ensemble(ensemble_tag)
    sig = world["sig"]
    strat = sig["strategy"].to_numpy()
    pols = ev.baseline_policies(sig)
    out: dict = {}

    held = (world["periods"]["val"][0], world["periods"]["oos"][1])
    train = world["periods"]["train"]

    # ---- A) paired stationary block bootstrap on the held-out year ----------
    names = ["rl", "always", "solo_s2", "solo_dmi", "solo_cpmt"]
    rets = {}
    for nm in names:
        p = pol if nm == "rl" else pols[nm]
        rets[nm], _ = daily_returns(world, held[0], held[1], p)
    n_days = len(rets["rl"])
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n_days, n_boot, mean_block, rng)
    boot = {nm: np.take(rets[nm], idx) for nm in names}          # (n_boot, n_days)
    sh = {nm: boot[nm].mean(1) / boot[nm].std(1) * np.sqrt(252) for nm in names}
    best_solo_sh = np.maximum.reduce([sh["solo_s2"], sh["solo_dmi"], sh["solo_cpmt"]])
    ret_tot = {nm: np.exp(boot[nm].sum(1)) - 1 for nm in names}
    best_solo_ret = np.maximum.reduce([ret_tot["solo_s2"], ret_tot["solo_dmi"], ret_tot["solo_cpmt"]])
    dd_rl = np.array([maxdd(boot["rl"][k]) for k in range(0, n_boot, max(1, n_boot // 2000))])
    out["bootstrap"] = dict(
        n=n_boot, days=n_days,
        rl_sharpe_ci=[float(np.percentile(sh["rl"], q)) for q in (5, 50, 95)],
        rl_ret_ci=[float(np.percentile(ret_tot["rl"], q)) for q in (5, 50, 95)],
        rl_maxdd_ci=[float(np.percentile(dd_rl, q)) for q in (5, 50, 95)],
        p_sharpe_beats_best_solo=float((sh["rl"] > best_solo_sh).mean()),
        p_ret_beats_best_solo=float((ret_tot["rl"] > best_solo_ret).mean()),
        p_sharpe_beats_always=float((sh["rl"] > sh["always"]).mean()),
        p_ret_beats_always=float((ret_tot["rl"] > ret_tot["always"]).mean()),
        sharpe_diff_ci=[float(np.percentile(sh["rl"] - best_solo_sh, q)) for q in (5, 50, 95)],
    )

    # ---- B) action-matched random nulls: held-out vs train ------------------
    res_null = {}
    for tag, (i0, i1), n_null in (("held", held, n_null_held), ("train", train, n_null_train)):
        freq = action_frequencies(world, pol, i0, i1)
        true_m = ev.eval_range(world, i0, i1, pol)
        null_sh = np.empty(n_null)
        null_net = np.empty(n_null)
        for k in range(n_null):
            rp = matched_random_policy(strat, freq, seed=10_000 + k)
            m = ev.eval_range(world, i0, i1, rp)
            null_sh[k] = m["sharpe"]
            null_net[k] = m["net"]
        res_null[tag] = dict(
            n=n_null, freq={s: [round(float(x), 3) for x in f] for s, f in freq.items()},
            true_sharpe=true_m["sharpe"], true_net=true_m["net"],
            null_sharpe_mean=float(null_sh.mean()), null_sharpe_sd=float(null_sh.std()),
            pct_sharpe=float((true_m["sharpe"] > null_sh).mean()),
            pct_net=float((true_m["net"] > null_net).mean()),
            z_sharpe=float((true_m["sharpe"] - null_sh.mean()) / max(null_sh.std(), 1e-9)),
        )
    out["null"] = res_null

    # ---- C) deflated Sharpe on the held-out year ----------------------------
    looks = json.loads((config.CHECKPOINT_DIR / "oos_looks.json").read_text())
    n_trials = max(1, len(looks))
    # benchmark: the best solo's held-out-year Sharpe
    best_solo_year = max(sharpe(rets["solo_s2"]), sharpe(rets["solo_dmi"]),
                         sharpe(rets["solo_cpmt"]))
    out["dsr"] = deflated_sharpe(rets["rl"], best_solo_year, n_trials)
    out["dsr"]["n_trials"] = n_trials
    out["dsr"]["benchmark_sharpe"] = best_solo_year
    return out


if __name__ == "__main__":
    import sys
    basis = sys.argv[1] if len(sys.argv) > 1 else "tv"
    tag = sys.argv[2] if len(sys.argv) > 2 else "deploy"
    out = run(basis=basis, ensemble_tag=tag)
    b = out["bootstrap"]
    print(f"A) Paired stationary block bootstrap, held-out year ({b['days']} days, {b['n']:,} draws)")
    print(f"   RL Sharpe 5/50/95%: {[round(x,2) for x in b['rl_sharpe_ci']]}   "
          f"return: {[f'{x:.1%}' for x in b['rl_ret_ci']]}   maxDD: {[f'{x:.1%}' for x in b['rl_maxdd_ci']]}")
    print(f"   P(RL Sharpe > best solo) = {b['p_sharpe_beats_best_solo']:.1%}   "
          f"P(RL return > best solo) = {b['p_ret_beats_best_solo']:.1%}")
    print(f"   P(RL Sharpe > always)    = {b['p_sharpe_beats_always']:.1%}   "
          f"P(RL return > always)    = {b['p_ret_beats_always']:.1%}")
    print(f"   (RL - best solo) Sharpe diff 5/50/95%: {[round(x,2) for x in b['sharpe_diff_ci']]}")
    for tag in ("held", "train"):
        n = out["null"][tag]
        print(f"\nB) Action-matched random null ({tag}, {n['n']:,} draws; per-strategy action freqs {n['freq']})")
        print(f"   true Sharpe {n['true_sharpe']:.3f} vs null {n['null_sharpe_mean']:.3f}±{n['null_sharpe_sd']:.3f} "
              f"-> percentile {n['pct_sharpe']:.1%}, z={n['z_sharpe']:.2f} | net percentile {n['pct_net']:.1%}")
    d = out["dsr"]
    print(f"\nC) Deflated Sharpe (held-out year, trials={d['n_trials']}, benchmark=best solo {d['benchmark_sharpe']:.2f})")
    print(f"   RL annual SR {d['sr_annual']:.2f} vs deflated hurdle {d['sr0_annual']:.2f} "
          f"-> P(skill) = {d['dsr_prob']:.1%}  (skew {d['skew']:.2f}, kurt {d['kurt']:.1f}, n={d['n_days']})")
