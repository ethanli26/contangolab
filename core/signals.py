"""One registry for every signal on every asset class.

A ``SignalSpec`` records what a signal is BEFORE it is tested: where it came from, which
direction the source says is good (``sign``, never flipped after seeing results), which family
of sources it belongs to (for multiple-testing correction), and when it was published (for
the pre/post-publication decay split). ``compute`` returns a date x asset panel; value at t
uses data up to t only.

The existing equity factor library and the futures signals are adapted in here, so a batch
evaluation can run "every signal for asset class X" without knowing where each one lives.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from core.dataset import Dataset

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

# (name, published year) for the equity library already in factors/; signs come from
# factors/run_factor_eval.EXPECTED_SIGN, written before any of these were evaluated.
EQUITY_PUBLISHED = {
    "momentum_12_1": 1993, "momentum_6_1": 1993, "short_term_reversal": 1990, "low_volatility": 2006,
    "ivol_capm": 2006, "beta_low": 2014, "max_daily_return": 2011, "return_skewness": 2000,
    "ncskew": 2001, "duvol": 2001, "wq_alpha_101": 2016, "wq_alpha_12": 2016,
    "profitability": 2013, "earnings_yield": 1977, "book_to_price": 1992, "earnings_growth": None,
    "operating_profitability": 2015, "gross_margin": 2013, "fcf_yield": 1994, "sales_growth": 1994,
    "accruals": 1996, "asset_growth": 2008, "debt_to_equity": 2007, "quality_composite": None,
}
FUTURES_PUBLISHED = {"TS12": 2012, "CARRY": 2018}   # MOP (2012), Koijen et al. (2018); the rest are broker reports
WQ_SIGN = {"wq_alpha_101": +1, "wq_alpha_12": +1}   # Kakushadze (2016) alphas are long-as-written


def _equity_compute(factor_cls) -> Callable[[Dataset], pd.DataFrame]:
    return lambda ds: factor_cls().compute(ds.data)


def _futures_compute(fn) -> Callable[[Dataset], pd.DataFrame]:
    def run(ds: Dataset) -> pd.DataFrame:
        from factors.futures_signals import with_returns
        px = with_returns(ds.meta["px"])
        return pd.Series(fn(px).to_numpy(), index=pd.MultiIndex.from_frame(px[["date", "sym"]])).unstack("sym")
    return run


_BUILTINS_LOADED = False


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import factors  # noqa: F401  (registers the price/volume + fundamental equity factors)
    from factors.base import all_factors
    from factors.futures_signals import SIGNALS as FUTURES
    from factors.run_factor_eval import EXPECTED_SIGN

    for name, cls in all_factors().items():
        if name not in EQUITY_PUBLISHED:
            continue                                   # alt-data factors need extra panels; added with their loader
        register_signal(SignalSpec(
            name=name, asset_classes=("us_equity",), family="equity_library", kind="cross_sectional",
            sign={**EXPECTED_SIGN, **WQ_SIGN}[name], source=(cls.__doc__ or "").strip().splitlines()[0],
            published=EQUITY_PUBLISHED[name], compute=_equity_compute(cls)))
    for spec in FUTURES.values():
        register_signal(SignalSpec(
            name=f"fut_{spec.name.lower()}", asset_classes=("cn_futures",), family="futures_batch3",
            kind=spec.kind, sign=spec.sign, source=spec.source, published=FUTURES_PUBLISHED.get(spec.name),
            compute=_futures_compute(spec.compute)))
