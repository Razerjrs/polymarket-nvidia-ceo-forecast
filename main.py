#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry: per-market adaptive volatility forecast.

Model (tuned on the 6 practice cases; practice mean pinball loss 0.265 vs the
house baseline's 0.364):
  - sigma: EWMA (half-life 240 min) of |1-min returns| over the last 720 min,
    scaled by 1.2533 (normal mean-|return| -> sd), clipped to [0.0004, 0.06].
    Robust to the jump-dominated return distribution (plain sd is far worse).
  - spread(h) = sigma * h**0.35 (sub-sqrt growth: moves saturate with horizon).
  - median(h) = x * (1 - 0.05 * (1 - exp(-h/240))): a slow multiplicative bleed
    toward 0 (long-shot prices decay as the over-round comes out).
  - quantiles: median + z_tau * spread with asymmetric widths
    (outer z x1.45, lower-inner x1.3, upper-inner x0.7), clamped to [0, 1].
  - tested and REJECTED (all worse on practice): momentum/trend extrapolation
    (any coefficient), mean-reversion pull, cross-market sum normalization,
    empirical h-step return quantiles, two-piece horizon growth.
No network, stdlib only, deterministic: same input -> byte-identical output.
"""
import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

Z = (-1.644854 * 1.45, -0.67449 * 1.3, 0.0, 0.67449 * 0.7, 1.644854 * 1.45)
WINDOW = 720          # minutes of history used for volatility
HALF_LIFE = 240.0     # EWMA half-life in minutes
SIGMA_FLOOR = 0.0004
SIGMA_CAP = 0.06
SPREAD_SCALE = 1.0
ALPHA = 0.35          # spread grows as h**ALPHA
DECAY = 0.05          # median bleed toward 0
TAU = 240.0           # bleed time constant in minutes


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def market_sigma(prices):
    """EWMA volatility of 1-min returns over the last WINDOW minutes."""
    rets = [abs(prices[i + 1] - prices[i]) for i in range(len(prices) - 1)]
    rets = rets[-WINDOW:]
    if not rets:
        return SIGMA_FLOOR
    a = 1.0 - 0.5 ** (1.0 / HALF_LIFE)
    wsum = 0.0
    vsum = 0.0
    w = 1.0
    for r in reversed(rets):
        vsum += w * r
        wsum += w
        w *= (1.0 - a)
    sigma = (vsum / wsum) * 1.2533
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast(prices, horizon):
    x = prices[-1] if prices else 0.5
    sigma = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = SPREAD_SCALE * sigma * (h ** ALPHA)
        median = x * (1.0 - DECAY * (1.0 - math.exp(-h / TAU)))
        row = [price6(median + z * spread) for z in Z]
        row.sort()
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        cid = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecasts = [
            {"id": m["id"], "q": forecast(m["prices"], case["horizon_minutes"])}
            for m in case["markets"]
        ]
        with open(os.path.join(OUTPUT_DIR, cid + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
