"""The platform's core drivers, shared by every asset class (BLUEPRINT 11, layers 1-3 and 8).

  dataset   one interface per asset class: panels, point-in-time universe, costs, labels
  signals   one registry for every factor/signal: source, sign, family, publication year
  evaluate  one monthly test for every signal on every dataset: IC + cost-aware long-short
  trials    an append-only log of every evaluation; multiple-testing math reads from it

A new asset class (A-shares next) only needs a dataset loader. A new factor only needs a
SignalSpec. Everything downstream (evaluation, trial log, alpha model, portfolio) is shared.
"""
