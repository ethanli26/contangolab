"""Datasets: one per market, all exposing the same shape.

A Dataset is everything an evaluation needs and nothing more:

  * ``data``: ``FactorData`` panels (date x market), point-in-time.
  * ``eligible``: date x asset bool mask of what was investable on each date, decided with data
    up to that date only.
  * trading assumptions: cost per side, number of quantile groups for long-short tests.

Loaders are registered by name, so ``load_dataset("cn_futures")`` and ``load_dataset("us_futures")``
look the same to everything downstream.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from contangolab import config


@dataclass
class FactorData:
    """Aligned date x market panels of the continuous contracts. ``extra`` holds the futures-specific
    panels (roll yield ``ry``, next-expiry return ``farRet``, liquid universe ``uni``, ...)."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    extra: dict[str, pd.DataFrame] | None = None


@dataclass
class Dataset:
    asset_class: str
    data: FactorData
    eligible: pd.DataFrame
    cost_bps_per_side: float
    groups: int                      # quantile groups for the long-short test (10 = deciles)
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def close(self) -> pd.DataFrame:
        return self.data.close


_LOADERS: dict[str, Callable[..., Dataset]] = {}


def register_loader(asset_class: str):
    def wrap(fn: Callable[..., Dataset]) -> Callable[..., Dataset]:
        _LOADERS[asset_class] = fn
        return fn
    return wrap


def asset_classes() -> list[str]:
    return list(_LOADERS)


def load_dataset(asset_class: str, **kwargs) -> Dataset:
    if asset_class not in _LOADERS:
        raise KeyError(f"no loader for {asset_class!r}; registered: {', '.join(_LOADERS)}")
    return _LOADERS[asset_class](**kwargs)


@register_loader("cn_futures")
def load_cn_futures(source_dir: str | None = None) -> Dataset:
    """China commodity futures, continuous contracts (see data/futures_cn.py)."""
    from contangolab.markets.china import load_continuous, to_factor_data

    px = load_continuous(source_dir)
    data, eligible, extra = to_factor_data(px)
    data.extra = {**extra, "uni": eligible}
    return Dataset("cn_futures", data, eligible, cost_bps_per_side=config.FUTURES_COST_BPS_PER_SIDE, groups=5,
                   description="China commodity futures (SHFE/DCE/CZCE/INE/GFEX), main-contract continuous, "
                               "liquid universe", meta={"px": px})


@register_loader("us_futures")
def load_us_futures(source_dir: str | None = None) -> Dataset:
    """US commodity futures (CME Group), continuous by the China rules, plus CFTC hedging pressure."""
    from contangolab.markets.cftc import attach, load_cot
    from contangolab.markets.china import to_factor_data
    from contangolab.markets.cme import load_continuous

    px = attach(load_continuous(source_dir), load_cot())
    data, eligible, extra = to_factor_data(px)
    data.extra = {**extra, "uni": eligible}
    return Dataset("us_futures", data, eligible, cost_bps_per_side=config.FUTURES_COST_BPS_PER_SIDE, groups=5,
                   description="US commodity futures (NYMEX/COMEX/CBOT/CME), main-contract continuous, "
                               "liquid universe", meta={"px": px})
