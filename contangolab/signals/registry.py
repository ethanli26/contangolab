"""One registry for every signal on every market.

A ``SignalSpec`` records what a signal is BEFORE it is tested: where it came from, which
direction the source says is good (``sign``, never flipped after seeing results), which family
of sources it belongs to (for multiple-testing correction), and when it was published (for
the pre/post-publication decay split). ``compute`` returns a date x asset panel; value at t
uses data up to t only.

The signal library (``contangolab.signals.library``) is adapted in here, so a batch evaluation can
run "every signal for market X" without knowing where each one lives. China signals are named
``fut_*`` and US signals ``usf_*``; those names are the keys in ``results/trials.jsonl``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from contangolab.engine.dataset import Dataset

KINDS = ("cross_sectional", "time_series")


@dataclass(frozen=True)
class SignalSpec:
    name: str
    asset_classes: tuple[str, ...]
    family: str                      # source family, the unit of multiple-testing correction
    kind: str                        # "cross_sectional" or "time_series"
    sign: int                        # +1: high value is good, -1: low value is good (from the source)
    source: str
    published: int | None            # year the source was published; None if not a published result
    compute: Callable[[Dataset], pd.DataFrame]

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")
        if self.sign not in (1, -1):
            raise ValueError(f"{self.name}: sign must be +1 or -1")


_SIGNALS: dict[str, SignalSpec] = {}


def register_signal(spec: SignalSpec) -> SignalSpec:
    if spec.name in _SIGNALS:
        raise ValueError(f"signal {spec.name!r} is already registered")
    _SIGNALS[spec.name] = spec
    return spec


def signals(asset_class: str | None = None, family: str | None = None) -> list[SignalSpec]:
    """Registered signals, optionally filtered by asset class and/or family (registration order)."""
    _ensure_builtins()
    return [s for s in _SIGNALS.values()
            if (asset_class is None or asset_class in s.asset_classes) and (family is None or s.family == family)]


def get_signal(name: str) -> SignalSpec:
    _ensure_builtins()
    return _SIGNALS[name]


# --- adapters for the existing libraries ------------------------------------------------------

FUTURES_PUBLISHED = {"TS12": 2012, "CARRY": 2018,   # MOP (2012), Koijen et al. (2018); the rest are broker reports
                     "BASISMOM": 2019, "HEDGEP": 2013}


def _futures_compute(fn) -> Callable[[Dataset], pd.DataFrame]:
    def run(ds: Dataset) -> pd.DataFrame:
        from contangolab.signals.library import with_returns
        px = with_returns(ds.meta["px"])
        return pd.Series(fn(px).to_numpy(), index=pd.MultiIndex.from_frame(px[["date", "sym"]])).unstack("sym")
    return run


_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from contangolab.signals.library import SIGNALS as FUTURES
    from contangolab.signals.library import US_SIGNALS as US_FUTURES

    for spec in FUTURES.values():
        register_signal(SignalSpec(
            name=f"fut_{spec.name.lower()}", asset_classes=("cn_futures",), family="futures_batch3",
            kind=spec.kind, sign=spec.sign, source=spec.source, published=FUTURES_PUBLISHED.get(spec.name),
            compute=_futures_compute(spec.compute)))
    for spec in US_FUTURES.values():
        register_signal(SignalSpec(
            name=f"usf_{spec.name.lower()}", asset_classes=("us_futures",), family="us_futures_c1",
            kind=spec.kind, sign=spec.sign, source=spec.source, published=FUTURES_PUBLISHED.get(spec.name),
            compute=_futures_compute(spec.compute)))
