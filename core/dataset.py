"""Datasets: one per asset class, all exposing the same shape.

A Dataset is everything an evaluation needs and nothing more:

  * ``data``: ``FactorData`` panels (date x asset), point-in-time.
  * ``eligible``: date x asset bool mask of what was investable on each date, decided with data
    up to that date only.
  * trading assumptions: cost per side, number of quantile groups for long-short tests.

Loaders are registered by asset-class name, so ``load_dataset("cn_futures")`` and, later,
``load_dataset("cn_equity")`` look the same to everything downstream.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

import config
from factors.base import FactorData


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


@register_loader("us_equity")
def load_us_equity(universe_mode: str | None = None, top_n: int | None = None) -> Dataset:
    """Survivorship-free US common stocks from the local Sharadar archive (see backtest/universe.py)."""
    from factors.run_factor_eval import SHARADAR_MAX_SYMBOLS, build_sharadar_factor_data

    data, eligible = build_sharadar_factor_data(universe_mode=universe_mode, top_n=top_n or SHARADAR_MAX_SYMBOLS)
    return Dataset("us_equity", data, eligible, cost_bps_per_side=config.VMM_COST_BPS_PER_SIDE, groups=10,
                   description=f"US common stocks, point-in-time top {top_n or SHARADAR_MAX_SYMBOLS} by liquidity, "
                               "delisted included (Sharadar)")


@register_loader("cn_futures")
def load_cn_futures(source_dir: str | None = None) -> Dataset:
    """China commodity futures, continuous contracts (see data/futures_cn.py)."""
    from data.futures_cn import load_continuous, to_factor_data

    px = load_continuous(source_dir)
    data, eligible, extra = to_factor_data(px)
    data.extra = {**extra, "uni": eligible}
    return Dataset("cn_futures", data, eligible, cost_bps_per_side=config.FUTURES_COST_BPS_PER_SIDE, groups=5,
                   description="China commodity futures (SHFE/DCE/CZCE/INE/GFEX), main-contract continuous, "
                               "liquid universe", meta={"px": px})
