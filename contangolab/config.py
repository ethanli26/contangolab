"""Paths and trading assumptions, overridable from a local .env (git-ignored, never committed)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(os.getenv("CONTANGOLAB_CACHE", REPO / "data" / "cache"))   # licensed data stays here, untracked
RESULTS = REPO / "results"                                             # committed: charts, tables, trial log
OUTPUT = REPO / "output"                                               # untracked scratch output

# China: folder with futures_daily.csv, contract_specs.csv, receipts.csv (written by the ~/ddb pull scripts).
FUTURES_CN_DIR = os.getenv("FUTURES_CN_DIR", str(CACHE / "futures_cn"))
# US: Databento GLBX.MDP3 statistics + daily bars (contangolab/markets/databento.py), CFTC reports.
FUTURES_US_DIR = os.getenv("FUTURES_US_DIR", str(CACHE / "futures_us"))
CFTC_DIR = os.getenv("CFTC_DIR", str(CACHE / "cftc"))
EIA_DIR = os.getenv("EIA_DIR", str(CACHE / "eia"))

FUTURES_COST_BPS_PER_SIDE = float(os.getenv("FUTURES_COST_BPS_PER_SIDE", "3"))  # ~1bp fee + ~1 tick, liquid contracts
