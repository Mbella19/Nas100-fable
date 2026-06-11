"""Success-gate evaluation, negative controls, and the final report."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config
from . import evaluate as ev
from .env import SignalEnv
from .ppo import EnsemblePolicy

A_FULL = 2
OOS_LOOK_FILE = config.CHECKPOINT_DIR / "oos_looks.json"


def log_oos_look(tag: str) -> int:
    looks = []
    if OOS_LOOK_FILE.exists():
        looks = json.loads(OOS_LOOK_FILE.read_text())
    looks.append(dict(tag=tag, ts=pd.Timestamp.utcnow().isoformat()))
    OOS_LOOK_FILE.write_text(json.dumps(looks, indent=2))
    return len(looks)


def gate_check(world: dict, policy, periods=("train", "val", "oos"),
               include_oos: bool = False) -> pd.DataFrame:
    """RL vs solos vs always on each period. Gate: net AND sharpe > best solo."""
    pols = ev.baseline_policies(world["sig"])
    rows = []
    use = [p for p in periods if include_oos or p != "oos"]
    for period in use:
        res = {}
        for name in ("solo_s2", "solo_dmi", "solo_cpmt", "always"):
            res[name] = ev.eval_policy(world, period, pols[name])
        res["rl"] = ev.eval_policy(world, period, policy)
        best_net = max(res[s]["net"] for s in ("solo_s2", "solo_dmi", "solo_cpmt"))
        best_sharpe = max(res[s]["sharpe"] for s in ("solo_s2", "solo_dmi", "solo_cpmt"))
        for name, m in res.items():
            rows.append(dict(period=period, policy=name, net=m["net"], sharpe=m["sharpe"],
                             maxdd=m["maxdd"], calmar=m["calmar"], trades=m["n_trades"],
                             win=m["win_rate"],
                             gate_net=(m["net"] > best_net) if name == "rl" else None,
                             gate_sharpe=(m["sharpe"] > best_sharpe) if name == "rl" else None))
    return pd.DataFrame(rows)


def gate_passed(table: pd.DataFrame, periods) -> bool:
    rl = table[table["policy"] == "rl"]
    rl = rl[rl["period"].isin(periods)]
    return bool(rl["gate_net"].all() and rl["gate_sharpe"].all())


def action_profile(world: dict, policy, period: str) -> dict:
    env = SignalEnv(world["sig"], world["feats"], world["mark"])
    i0, i1 = world["periods"][period]
    obs = env.reset(i0, i1)
    acts = []
    done = False
    while not done:
        a = policy(obs, env.i)
        acts.append((world["sig"]["strategy"].iloc[env.i], a))
        obs, _, done, _ = env.step(a)
    df = pd.DataFrame(acts, columns=["strategy", "action"])
    mults = config.RL["action_multipliers"]
    prof = df.groupby("strategy")["action"].agg(
        take_rate=lambda s: float((s > 0).mean()),
        avg_mult=lambda s: float(np.mean([mults[a] for a in s])))
    return dict(profile=prof, take_rate=float((df["action"] > 0).mean()),
                counts=Counter(df["action"].tolist()))


def negative_controls(world: dict, policy, period: str = "val",
                      n_random: int = 20, n_perm: int = 20) -> dict:
    """(a) random gate at the policy's take-rate (distribution over seeds);
    (b) policy on PERMUTED signal features — a proper null over n_perm draws,
        not a single permutation (a single draw once landed 2 sigma high and
        misread as "features do nothing"). Note the 4 portfolio-state features
        are appended after the feature matrix in SignalEnv._obs and are NOT
        permuted, so (b) tests only the per-signal feature channel;
    (c) channel ablations close that blind spot: zero the signal features
        (z-scored, so 0 = train mean) keeping real portfolio state, and the
        reverse — which input channel carries the edge?"""
    prof = action_profile(world, policy, period)
    p_take = prof["take_rate"]
    pols = ev.baseline_policies(world["sig"])
    rnd = [ev.eval_policy(world, period, pols["random_gate"](p_take, seed=s))
           for s in range(n_random)]
    rnd_net = float(np.mean([m["net"] for m in rnd]))
    rnd_sh = np.array([m["sharpe"] for m in rnd])

    perm_net, perm_sh = [], []
    for s in range(n_perm):
        rng = np.random.default_rng(s)
        w2 = dict(world, feats=world["feats"][rng.permutation(len(world["feats"]))])
        m = ev.eval_policy(w2, period, policy)
        perm_net.append(m["net"])
        perm_sh.append(m["sharpe"])
    perm_sh = np.array(perm_sh)

    nf = len(world["names"])

    def masked(fn):
        def p(obs, i):
            o = obs.copy()
            fn(o)
            return policy(o, i)
        return p

    m_rl = ev.eval_policy(world, period, policy)
    m_sig_only = ev.eval_policy(world, period,
                                masked(lambda o: o.__setitem__(slice(nf, None), 0.0)))
    m_port_only = ev.eval_policy(world, period,
                                 masked(lambda o: o.__setitem__(slice(0, nf), 0.0)))
    z = float((m_rl["sharpe"] - perm_sh.mean()) / max(perm_sh.std(), 1e-9))
    return dict(take_rate=p_take,
                rl=(m_rl["net"], m_rl["sharpe"]),
                random=(rnd_net, float(rnd_sh.mean()), float(rnd_sh.std())),
                permuted=(float(np.mean(perm_net)), float(perm_sh.mean()),
                          float(perm_sh.std()), z),
                signal_only=(m_sig_only["net"], m_sig_only["sharpe"]),
                port_only=(m_port_only["net"], m_port_only["sharpe"]))


def seed_dispersion(world: dict, models: list, period: str) -> pd.DataFrame:
    rows = []
    for k, m in enumerate(models):
        pol = EnsemblePolicy([m])
        r = ev.eval_policy(world, period, pol)
        rows.append(dict(seed=k, net=r["net"], sharpe=r["sharpe"], maxdd=r["maxdd"]))
    return pd.DataFrame(rows)


def write_final_report(world: dict, policy, models: list, grid_info: dict,
                       path: str = None) -> str:
    """Full evaluation incl. the locked-OOS look (logged) and the report file."""
    n_looks = log_oos_look("final_report")
    table = gate_check(world, policy, include_oos=True)
    passed = gate_passed(table, ("train", "val", "oos"))
    nc = negative_controls(world, policy, "val")
    disp = {p: seed_dispersion(world, models, p) for p in ("train", "val", "oos")}
    profs = {p: action_profile(world, policy, p) for p in ("train", "val", "oos")}
    png = str(config.REPORTS_DIR / "rl_equity_curves.png")
    plot_curves(world, policy, png)

    def fmt(t: pd.DataFrame) -> str:
        t = t.copy()
        t["net"] = t["net"].map(lambda x: f"{x:,.0f}")
        for c in ("sharpe", "calmar"):
            t[c] = t[c].round(2)
        t["maxdd"] = (t["maxdd"] * 100).round(1)
        t["win"] = (t["win"] * 100).round(1)
        return t.drop(columns=["gate_net", "gate_sharpe"]).to_markdown(index=False)

    lines = ["# NAS100 RL Meta-Controller — Final Report", ""]
    lines.append(f"**Success gate (net profit AND Sharpe > best individual strategy, "
                 f"all three periods): {'PASSED' if passed else 'FAILED'}**")
    lines.append("")
    lines.append(f"Grid-selected config: `{grid_info}`. Ensemble of {len(models)} seeds, "
                 f"deterministic mean-logits policy. Locked-OOS evaluations so far: {n_looks}.")
    for period in ("train", "val", "oos"):
        sub = table[table["period"] == period]
        rl = sub[sub["policy"] == "rl"].iloc[0]
        lines.append(f"\n## {period}\n")
        lines.append(fmt(sub))
        lines.append(f"\nGate: net {'PASS' if rl['gate_net'] else 'FAIL'}, "
                     f"sharpe {'PASS' if rl['gate_sharpe'] else 'FAIL'}")
        pr = profs[period]
        lines.append(f"\nTake-rate {pr['take_rate']:.1%}; per-strategy profile:\n")
        lines.append(pr["profile"].round(3).to_markdown())
        lines.append(f"\nSeed dispersion:\n")
        lines.append(disp[period].round(2).to_markdown(index=False))
    lines.append("\n## Negative controls (validation period)\n")
    lines.append(f"- RL: net {nc['rl'][0]:,.0f}, Sharpe {nc['rl'][1]:.2f}")
    lines.append(f"- Random gate @ same take-rate ({nc['take_rate']:.1%}, 20 seeds): "
                 f"net {nc['random'][0]:,.0f}, Sharpe {nc['random'][1]:.2f} ± {nc['random'][2]:.2f}")
    lines.append(f"- Permutation null (20 draws, signal features shuffled, portfolio "
                 f"state real): net {nc['permuted'][0]:,.0f}, Sharpe {nc['permuted'][1]:.2f} "
                 f"± {nc['permuted'][2]:.2f} -> RL z = {nc['permuted'][3]:.2f}")
    lines.append(f"- Ablation, signal features only (portfolio state zeroed): "
                 f"net {nc['signal_only'][0]:,.0f}, Sharpe {nc['signal_only'][1]:.2f}")
    lines.append(f"- Ablation, portfolio state only (signal features zeroed): "
                 f"net {nc['port_only'][0]:,.0f}, Sharpe {nc['port_only'][1]:.2f}")
    lines.append(f"\n![equity curves](rl_equity_curves.png)\n")
    out = "\n".join(lines)
    p = path or str(config.REPORTS_DIR / "final_report.md")
    with open(p, "w") as f:
        f.write(out)
    return out


def plot_curves(world: dict, policy, path: str) -> None:
    pols = ev.baseline_policies(world["sig"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, period in zip(axes, ("train", "val", "oos")):
        for name, pol, color in (("RL ensemble", policy, "#d62728"),
                                 ("always-take", pols["always"], "#7f7f7f"),
                                 ("S2 solo", pols["solo_s2"], "#1f77b4"),
                                 ("DMI solo", pols["solo_dmi"], "#2ca02c"),
                                 ("CPMT solo", pols["solo_cpmt"], "#9467bd")):
            m = ev.eval_policy(world, period, pol)
            c = m["curve"]
            ax.plot(c["srv_day"], c["equity"] / 1000, label=name, lw=1.2, color=color,
                    alpha=0.95 if name == "RL ensemble" else 0.7)
        ax.set_title(period)
        ax.grid(alpha=0.3)
        ax.set_ylabel("equity ($k)")
        if period == "train":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
