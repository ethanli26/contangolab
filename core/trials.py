"""Trial registry: an append-only log of every signal evaluation (BLUEPRINT 11, P4).

Testing hundreds of factors guarantees some "significant" results by luck. The defence is to
count every test, including the failures and the reruns, and to correct for that count. This
log is where the count lives:

  * ``log_trial`` appends one JSON line per evaluation to ``research/trials.jsonl`` (committed,
    so the history of what was tried is auditable in git), stamped with time, git commit and
    the protocol it ran under.
  * ``correction_table`` takes the latest result per (signal, asset class, protocol) and adds
    one-sided p-values in the source's predicted direction plus Benjamini-Hochberg q-values,
    within each family and across everything.
  * ``survivors`` applies the pre-registered survivor rule (BLUEPRINT 11, P1b).
  * ``search_summary`` gives the Deflated Sharpe Ratio its inputs (number of trials and the
    spread of their Sharpes) from the log instead of by hand.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
TRIALS_PATH = REPO / "research" / "trials.jsonl"
PROTOCOL = "b11-p1b-v1"            # BLUEPRINT 11 P1b evaluation protocol, as pre-registered 2026-10-10
SURVIVOR_Q = 0.10


def _git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", f":!{TRIALS_PATH.relative_to(REPO)}"],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")      # the log itself changing is not a dirty tree
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_trial(result: dict, protocol: str = PROTOCOL, path: Path = TRIALS_PATH, note: str = "") -> dict:
    """Append one evaluation result (a ``core.evaluate.summarize`` dict) to the registry."""
    record = {"logged_at": datetime.now(UTC).isoformat(timespec="seconds"), "protocol": protocol,
              "commit": _git_commit(), "note": note, **result}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=_jsonable) + "\n")
    return record


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    raise TypeError(f"not JSON serializable: {type(x)}")


def load_trials(path: Path = TRIALS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with open(path) as fh:
        return pd.DataFrame([json.loads(line) for line in fh if line.strip()])


def one_sided_p(t_stat: float | None, n: int | None) -> float:
    """P(t >= observed) under no effect, in the source's predicted direction (t-dist, n-1 df)."""
    if t_stat is None or n is None or n < 3 or not np.isfinite(t_stat):
        return 1.0
    return float(stats.t.sf(t_stat, df=n - 1))


def bh_qvalues(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values), same index as ``p``."""
    p = p.astype(float)
    m = len(p)
    if m == 0:
        return p
    order = p.sort_values()
    ranked = order * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1.0)
    return pd.Series(q.to_numpy(), index=order.index).reindex(p.index)


def latest(trials: pd.DataFrame, protocol: str | None = PROTOCOL) -> pd.DataFrame:
    """The most recent trial per (signal, asset class) under ``protocol`` (None = all protocols)."""
    if trials.empty:
        return trials
    t = trials if protocol is None else trials[trials["protocol"] == protocol]
    return (t.sort_values("logged_at", kind="stable")
             .drop_duplicates(["signal", "asset_class"], keep="last").reset_index(drop=True))


def correction_table(trials: pd.DataFrame, protocol: str | None = PROTOCOL) -> pd.DataFrame:
    """Latest result per signal with one-sided p, BH q within family and overall, and survivor flag."""
    t = latest(trials, protocol).copy()
    if t.empty:
        return t
    t["p"] = [one_sided_p(ts, n) for ts, n in zip(t["t_stat"], t["months"])]
    t["q_family"] = t.groupby("family")["p"].transform(bh_qvalues)
    t["q_all"] = bh_qvalues(t["p"])
    t["survivor"] = [_is_survivor(row) for row in t.to_dict("records")]
    return t.sort_values("p").reset_index(drop=True)


def _is_survivor(row: dict) -> bool:
    """BLUEPRINT 11 P1b: q_all <= 0.10, predicted direction confirmed, positive in both halves."""
    halves = [row.get("h1"), row.get("h2")]
    both_positive = all(isinstance(h, dict) and (h.get("ann_return") or 0) > 0 for h in halves)
    return bool(row["q_all"] <= SURVIVOR_Q and (row.get("t_stat") or 0) > 0 and both_positive)


def search_summary(trials: pd.DataFrame, asset_class: str | None = None, protocol: str | None = None) -> dict:
    """How hard the search was: distinct configurations tried and the spread of their Sharpes (DSR inputs)."""
    t = trials if asset_class is None else trials[trials["asset_class"] == asset_class]
    t = latest(t, protocol)
    sharpes = pd.to_numeric(t.get("sharpe", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"n_trials": int(len(t)), "sharpes": sharpes.tolist(),
            "sharpe_var": float(sharpes.var(ddof=1)) if len(sharpes) > 1 else 0.0}
