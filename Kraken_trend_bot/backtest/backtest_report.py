#!/usr/bin/env python
"""
backtest/backtest_report.py
===========================
Visual Backtest Report Generator for the Kraken Trend Bot.

Generates professional-grade charts from backtest results:
  * Equity curve with BTC buy-and-hold benchmark
  * Drawdown underwater chart
  * Trade P&L distribution
  * Per-symbol performance breakdown
  * Social-media-ready summary card (Twitter/X optimised 1200x675)

Usage:
    python -m backtest.backtest_report                            # latest result
    python -m backtest.backtest_report --file results/xxx.json    # specific file
    python -m backtest.backtest_report --all                      # every result
    python -m backtest.backtest_report --rerun                    # re-run backtest first
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patheffects as pe
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("ERROR: matplotlib is required.  Install with:  pip install matplotlib")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "backtest" / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# ─── Professional dark-theme palette ───────────────────────────────
C = {
    "bg":       "#0f1116",
    "panel":    "#1a1e2e",
    "text":     "#e2e8f0",
    "muted":    "#94a3b8",
    "dim":      "#64748b",
    "green":    "#22c55e",
    "red":      "#ef4444",
    "blue":     "#3b82f6",
    "purple":   "#a855f7",
    "orange":   "#f97316",
    "yellow":   "#eab308",
    "cyan":     "#06b6d4",
    "grid":     "#1e293b",
    "border":   "#334155",
    "accent":   "#818cf8",
}

SYMBOL_COLORS = [
    C["blue"], C["green"], C["purple"], C["orange"],
    C["cyan"], C["yellow"], C["red"], C["accent"],
]


def _apply_theme() -> None:
    """Apply the dark theme globally to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":   C["bg"],
        "axes.facecolor":     C["panel"],
        "axes.edgecolor":     C["border"],
        "axes.labelcolor":    C["text"],
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "text.color":         C["text"],
        "xtick.color":        C["muted"],
        "ytick.color":        C["muted"],
        "grid.color":         C["grid"],
        "grid.alpha":         0.4,
        "grid.linewidth":     0.5,
        "font.family":        ["DejaVu Sans", "sans-serif"],
        "font.size":          11,
        "legend.facecolor":   C["panel"],
        "legend.edgecolor":   C["border"],
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.3,
    })


# ═══════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════

def load_result(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_result() -> Optional[Path]:
    files = list(RESULTS_DIR.glob("backtest_*.json"))
    if not files:
        return None
    # Sort by file modification time (newest first)
    return max(files, key=lambda p: p.stat().st_mtime)


def find_all_results() -> List[Path]:
    return sorted(RESULTS_DIR.glob("backtest_*.json"), reverse=True)


def _load_btc_benchmark() -> Optional[List[Tuple[int, float]]]:
    """Load BTC/USD close prices from the OHLCV cache for benchmarking."""
    cache_dir = DATA_DIR / "cache_ohlcv"
    if not cache_dir.exists():
        return None
    for f in sorted(cache_dir.glob("BTC_USD__*__*d.json"), reverse=True):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            rows = raw.get("ohlcv") or raw
            if isinstance(rows, list) and len(rows) > 100:
                return [(int(r[0]), float(r[4])) for r in rows]
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════
# Advanced metric computation
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all performance metrics from backtest result data."""
    m: Dict[str, Any] = {
        "total_return_pct":  data.get("total_return_pct", 0.0),
        "max_drawdown_pct":  data.get("max_drawdown_pct", 0.0),
        "win_rate":          data.get("win_rate", 0.0),
        "profit_factor":     data.get("profit_factor"),
        "expectancy":        data.get("expectancy", 0.0),
        "total_trades":      data.get("total_trades", 0),
        "net_pnl":           data.get("net_pnl", 0.0),
        "total_fees":        data.get("total_fees", 0.0),
        "initial_capital":   data.get("initial_capital", 0.0),
        "final_equity":      data.get("final_equity", 0.0),
        "ml_accuracy":       data.get("ml_accuracy", 0.0),
    }

    # ── Sharpe / Sortino from equity series ──
    eq_series = data.get("equity_series", [])
    if len(eq_series) > 20:
        values = np.array([v for _, v in eq_series], dtype=np.float64)
        rets = np.diff(values) / values[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 1 and np.std(rets) > 1e-12:
            # Annualise assuming 96 bars/day for 15-min candles
            ann = np.sqrt(252 * 96)
            m["sharpe"] = float(np.mean(rets) / np.std(rets) * ann)
            neg = rets[rets < 0]
            if len(neg) > 0 and np.std(neg) > 1e-12:
                m["sortino"] = float(np.mean(rets) / np.std(neg) * ann)
            else:
                m["sortino"] = None
        else:
            m["sharpe"] = 0.0
            m["sortino"] = None
    else:
        m["sharpe"] = None
        m["sortino"] = None

    # ── Calmar ratio ──
    dd = m["max_drawdown_pct"]
    m["calmar"] = m["total_return_pct"] / dd if dd > 0.001 else None

    # ── Best / worst symbol ──
    per_sym = data.get("per_symbol", {})
    if per_sym:
        best = max(per_sym.items(), key=lambda x: x[1].get("net_pnl", 0))
        worst = min(per_sym.items(), key=lambda x: x[1].get("net_pnl", 0))
        m["best_symbol"] = best[0]
        m["best_pnl"] = best[1].get("net_pnl", 0)
        m["worst_symbol"] = worst[0]
        m["worst_pnl"] = worst[1].get("net_pnl", 0)
    else:
        m["best_symbol"] = "N/A"
        m["worst_symbol"] = "N/A"

    # ── Win / loss averages from trades list ──
    trades = data.get("trades", [])
    sells = [t for t in trades if t.get("side") == "sell"]
    if sells:
        pnls = [t.get("pnl", 0) for t in sells]
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]
        m["avg_win"] = float(np.mean(wins)) if wins else 0.0
        m["avg_loss"] = float(np.mean(losses)) if losses else 0.0
        m["largest_win"] = max(pnls) if pnls else 0.0
        m["largest_loss"] = min(pnls) if pnls else 0.0

    return m


# ═══════════════════════════════════════════════════════════════════
# Chart 1 — Equity Curve  (with optional BTC benchmark)
# ═══════════════════════════════════════════════════════════════════

def chart_equity_curve(
    data: Dict, metrics: Dict, save_path: Path,
) -> bool:
    eq_series = data.get("equity_series", [])
    if len(eq_series) < 5:
        return False

    _apply_theme()
    fig, ax = plt.subplots(figsize=(14, 6))

    timestamps = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in eq_series]
    values = [v for _, v in eq_series]
    initial = values[0] if values[0] != 0 else 1
    pct = [(v / initial - 1) * 100 for v in values]

    # Bot equity
    ax.plot(timestamps, pct, color=C["blue"], linewidth=1.8,
            label="Bot Equity", zorder=3)

    # Green / red fill
    ax.fill_between(timestamps, pct, 0,
                    where=[v >= 0 for v in pct],
                    color=C["green"], alpha=0.07, interpolate=True)
    ax.fill_between(timestamps, pct, 0,
                    where=[v < 0 for v in pct],
                    color=C["red"], alpha=0.07, interpolate=True)
    ax.axhline(0, color=C["muted"], linewidth=0.6, linestyle="--", alpha=0.5)

    # BTC benchmark overlay
    btc_data = _load_btc_benchmark()
    if btc_data:
        start_ts, end_ts = int(eq_series[0][0]), int(eq_series[-1][0])
        btc_filt = [(t, p) for t, p in btc_data if start_ts <= t <= end_ts]
        if len(btc_filt) > 10:
            btc_dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc)
                         for t, _ in btc_filt]
            btc_vals = [p for _, p in btc_filt]
            btc0 = btc_vals[0] if btc_vals[0] != 0 else 1
            btc_pct = [(v / btc0 - 1) * 100 for v in btc_vals]
            ax.plot(btc_dates, btc_pct, color=C["orange"], linewidth=1.2,
                    linestyle="--", alpha=0.7, label="BTC Buy & Hold")

    # End-value annotation
    final = pct[-1]
    colour = C["green"] if final >= 0 else C["red"]
    ax.annotate(
        f"{final:+.2f}%", xy=(timestamps[-1], final),
        fontsize=12, fontweight="bold", color=colour,
        xytext=(10, 10), textcoords="offset points",
        path_effects=[pe.withStroke(linewidth=3, foreground=C["bg"])],
    )

    # Stats box
    lines = [
        f"Return: {final:+.2f}%",
        f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"Trades: {metrics.get('total_trades', 0)}",
    ]
    if metrics.get("sharpe") is not None:
        lines.append(f"Sharpe: {metrics['sharpe']:.2f}")
    props = dict(boxstyle="round,pad=0.5", facecolor=C["panel"],
                 edgecolor=C["border"], alpha=0.9)
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment="top", bbox=props)

    ax.set_title("Portfolio Equity Curve", fontsize=16, pad=15)
    ax.set_ylabel("Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    fig.savefig(save_path, facecolor=C["bg"])
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════
# Chart 2 — Drawdown
# ═══════════════════════════════════════════════════════════════════

def chart_drawdown(data: Dict, save_path: Path) -> bool:
    eq_series = data.get("equity_series", [])
    if len(eq_series) < 5:
        return False

    _apply_theme()
    fig, ax = plt.subplots(figsize=(14, 4))

    timestamps = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                  for ts, _ in eq_series]
    values = np.array([v for _, v in eq_series], dtype=np.float64)
    peak = np.maximum.accumulate(values)
    dd_pct = np.where(peak > 0, (values - peak) / peak * 100, 0)

    ax.fill_between(timestamps, dd_pct, 0, color=C["red"], alpha=0.25)
    ax.plot(timestamps, dd_pct, color=C["red"], linewidth=1.2)

    # Max drawdown annotation
    idx = int(np.argmin(dd_pct))
    ax.annotate(
        f"Max DD: {dd_pct[idx]:.2f}%",
        xy=(timestamps[idx], dd_pct[idx]),
        fontsize=11, fontweight="bold", color=C["red"],
        xytext=(0, -25), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5),
        path_effects=[pe.withStroke(linewidth=3, foreground=C["bg"])],
    )

    ax.set_title("Drawdown (Underwater Equity)", fontsize=14, pad=10)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=30)

    fig.savefig(save_path, facecolor=C["bg"])
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════
# Chart 3 — Trade P&L Distribution
# ═══════════════════════════════════════════════════════════════════

def chart_trade_distribution(data: Dict, save_path: Path) -> bool:
    trades = data.get("trades", [])
    sells = [t for t in trades if t.get("side") == "sell"]
    if len(sells) < 2:
        return False

    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]},
    )

    pnls = [t.get("pnl", 0) for t in sells]
    colors = [C["green"] if p >= 0 else C["red"] for p in pnls]

    # Left — per-trade P&L bar chart
    ax1.bar(range(len(pnls)), pnls, color=colors, alpha=0.85, width=0.8)
    ax1.axhline(0, color=C["muted"], linewidth=0.6, linestyle="--")
    ax1.set_title("Per-Trade P&L", fontsize=14, pad=10)
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("P&L ($)")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right — win / loss count
    n_wins = sum(1 for p in pnls if p >= 0)
    n_losses = sum(1 for p in pnls if p < 0)
    labels = ["Wins", "Losses"]
    counts = [n_wins, n_losses]
    bar_colors = [C["green"], C["red"]]

    bars = ax2.bar(labels, counts, color=bar_colors, alpha=0.85, width=0.55)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, str(cnt),
                 ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=C["text"])

    wr = n_wins / len(pnls) * 100 if pnls else 0
    ax2.set_title(f"Win / Loss  (WR: {wr:.1f}%)", fontsize=14, pad=10)
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, facecolor=C["bg"])
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════
# Chart 4 — Per-Symbol Breakdown
# ═══════════════════════════════════════════════════════════════════

def chart_symbol_breakdown(data: Dict, save_path: Path) -> bool:
    per_sym = data.get("per_symbol", {})
    if not per_sym:
        return False

    _apply_theme()
    n = len(per_sym)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.9 + 2)))

    sorted_syms = sorted(per_sym.items(), key=lambda x: x[1].get("net_pnl", 0))
    names = [s for s, _ in sorted_syms]
    pnls = [v.get("net_pnl", 0) for _, v in sorted_syms]
    trade_counts = [v.get("trades", 0) for _, v in sorted_syms]
    win_rates = [v.get("win_rate", 0) for _, v in sorted_syms]
    colors = [C["green"] if p >= 0 else C["red"] for p in pnls]

    bars = ax.barh(names, pnls, color=colors, alpha=0.85, height=0.55)

    for bar, pnl, tc, wr in zip(bars, pnls, trade_counts, win_rates):
        offset = 0.001 if pnl >= 0 else -0.001
        ha = "left" if pnl >= 0 else "right"
        label = f"${pnl:+.2f}  ({tc}t, {wr:.0f}% WR)"
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            label, ha=ha, va="center", fontsize=10, color=C["text"],
            path_effects=[pe.withStroke(linewidth=2, foreground=C["bg"])],
        )

    ax.axvline(0, color=C["muted"], linewidth=0.6, linestyle="--")
    ax.set_title("Performance by Symbol", fontsize=14, pad=15)
    ax.set_xlabel("Net P&L ($)")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(save_path, facecolor=C["bg"])
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════
# Chart 5 — Social-Media Summary Card  (Twitter/X: 1200 x 675)
# ═══════════════════════════════════════════════════════════════════

def chart_summary_card(data: Dict, metrics: Dict, save_path: Path) -> bool:
    _apply_theme()
    fig = plt.figure(figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor(C["bg"])

    # ── Header ──
    fig.text(0.50, 0.95, "KRAKEN TREND BOT",
             fontsize=24, fontweight="bold", color=C["accent"],
             ha="center", va="top", fontfamily="monospace")
    fig.text(0.50, 0.90, "BACKTEST PERFORMANCE REPORT",
             fontsize=13, color=C["muted"], ha="center", va="top",
             fontfamily="monospace", fontweight="bold")

    symbols = data.get("symbols", [])
    config = data.get("config", {})
    tf = config.get("timeframe", "15m")
    ml_str = "ML ON" if config.get("use_ml") else "ML OFF"
    fig.text(0.50, 0.855,
             f"{len(symbols)} Assets   |   {tf}   |   {ml_str}"
             f"   |   Fee {config.get('fee_pct', 0.26)}%",
             fontsize=10, color=C["dim"], ha="center")

    fig.add_artist(plt.Line2D(
        [0.08, 0.92], [0.83, 0.83],
        transform=fig.transFigure, color=C["border"], lw=1))

    # ── Metric block helper ──
    def draw_metric(cx: float, cy: float, label: str,
                    value_str: str, value_color: str = C["text"]) -> None:
        bw, bh = 0.24, 0.13
        box = FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.008",
            facecolor=C["panel"], edgecolor=C["border"], linewidth=0.8,
            transform=fig.transFigure, clip_on=False,
        )
        fig.patches.append(box)
        fig.text(cx, cy + 0.025, label,
                 fontsize=8, color=C["muted"], ha="center", va="center",
                 fontweight="bold", transform=fig.transFigure)
        fig.text(cx, cy - 0.018, value_str,
                 fontsize=20, color=value_color, ha="center", va="center",
                 fontweight="bold", transform=fig.transFigure)

    # Colour helpers
    ret = metrics["total_return_pct"]
    ret_c = C["green"] if ret >= 0 else C["red"]
    exp = metrics.get("expectancy", 0)
    exp_c = C["green"] if exp >= 0 else C["red"]
    pf = metrics.get("profit_factor")
    pf_s = f"{pf:.2f}" if pf and pf < 999 else "---"

    # Row 1  (y = 0.72)
    draw_metric(0.20, 0.72, "TOTAL RETURN",  f"{ret:+.2f}%",               ret_c)
    draw_metric(0.50, 0.72, "WIN RATE",      f"{metrics['win_rate']:.1f}%")
    draw_metric(0.80, 0.72, "PROFIT FACTOR", pf_s)

    # Row 2  (y = 0.52)
    draw_metric(0.20, 0.52, "MAX DRAWDOWN",
                f"{metrics['max_drawdown_pct']:.2f}%", C["orange"])
    draw_metric(0.50, 0.52, "TOTAL TRADES",
                f"{metrics['total_trades']}")
    draw_metric(0.80, 0.52, "EXPECTANCY",
                f"${exp:+.2f}", exp_c)

    # Row 3  (y = 0.32)
    draw_metric(0.20, 0.32, "TOTAL FEES",
                f"${metrics['total_fees']:.2f}")
    draw_metric(0.50, 0.32, "ML ACCURACY",
                f"{metrics['ml_accuracy']:.1f}%")
    draw_metric(0.80, 0.32, "BEST SYMBOL",
                metrics.get("best_symbol", "N/A"), C["cyan"])

    # ── Extra metrics line ──
    extras: List[str] = []
    if metrics.get("sharpe") is not None:
        extras.append(f"Sharpe: {metrics['sharpe']:.2f}")
    if metrics.get("sortino") is not None:
        extras.append(f"Sortino: {metrics['sortino']:.2f}")
    if metrics.get("calmar") is not None:
        extras.append(f"Calmar: {metrics['calmar']:.2f}")
    if extras:
        fig.text(0.50, 0.20, "   |   ".join(extras),
                 fontsize=10, color=C["muted"], ha="center")

    # ── Bottom bar ──
    fig.add_artist(plt.Line2D(
        [0.08, 0.92], [0.12, 0.12],
        transform=fig.transFigure, color=C["border"], lw=1))

    ema_f = config.get("ema_fast", 21)
    ema_s = config.get("ema_slow", 55)
    sl = config.get("stop_loss_pct", 15)
    fig.text(0.50, 0.075,
             f"EMA({ema_f}/{ema_s}) + RSI + ATR   |   "
             f"Stop Loss: {sl}%   |   Trailing Stop",
             fontsize=9, color=C["dim"], ha="center")
    fig.text(0.50, 0.030,
             f"Generated {datetime.now().strftime('%Y-%m-%d')}   |   "
             "Simulated results only — not financial advice",
             fontsize=8, color=C["dim"], ha="center", style="italic")

    fig.savefig(save_path, facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════════════
# Report orchestrator
# ═══════════════════════════════════════════════════════════════════

def generate_report(json_path: Path) -> Path:
    """Generate all available charts for one backtest result file."""
    print(f"\n  Loading: {json_path.name}")
    data = load_result(json_path)
    metrics = compute_metrics(data)

    tag = data.get("tag", "")
    ts = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    chart_dir = CHARTS_DIR / f"{tag}_{ts}" if tag else CHARTS_DIR / ts
    chart_dir.mkdir(parents=True, exist_ok=True)

    generated: List[str] = []
    has_equity = len(data.get("equity_series", [])) >= 5
    has_trades = len([t for t in data.get("trades", [])
                      if t.get("side") == "sell"]) >= 2

    # Always possible (uses summary data only)
    if chart_summary_card(data, metrics, chart_dir / "summary_card.png"):
        generated.append("summary_card.png")
        print("    [+] Summary card  (social media ready)")

    if chart_symbol_breakdown(data, chart_dir / "symbol_breakdown.png"):
        generated.append("symbol_breakdown.png")
        print("    [+] Symbol breakdown")

    # Require equity series
    if has_equity:
        if chart_equity_curve(data, metrics, chart_dir / "equity_curve.png"):
            generated.append("equity_curve.png")
            print("    [+] Equity curve  (with BTC benchmark)")
        if chart_drawdown(data, chart_dir / "drawdown.png"):
            generated.append("drawdown.png")
            print("    [+] Drawdown chart")
    else:
        print("    [ ] Equity curve / drawdown  --  "
              "re-run backtest to generate (see --rerun)")

    # Require trades list
    if has_trades:
        if chart_trade_distribution(data, chart_dir / "trade_pnl.png"):
            generated.append("trade_pnl.png")
            print("    [+] Trade P&L distribution")
    else:
        print("    [ ] Trade distribution  --  "
              "re-run backtest to generate (see --rerun)")

    # ── Print summary ──
    print(f"\n  {'─' * 50}")
    ret = metrics["total_return_pct"]
    ret_icon = "+" if ret >= 0 else "-"
    print(f"    Total Return:    {ret:+.2f}%")
    print(f"    Win Rate:        {metrics['win_rate']:.1f}%")
    pf = metrics.get("profit_factor")
    pf_str = f"{pf:.2f}" if pf is not None and 0 < pf < 999 else ("---" if pf == 0 or pf is None else "inf")
    print(f"    Profit Factor:   {pf_str}")
    print(f"    Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
    print(f"    Total Trades:    {metrics['total_trades']}")
    if metrics.get("sharpe") is not None:
        print(f"    Sharpe Ratio:    {metrics['sharpe']:.2f}")
    print(f"  {'─' * 50}")
    print(f"\n    Charts saved to: {chart_dir}")
    print(f"    Files: {', '.join(generated)}")

    return chart_dir


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual backtest reports for Kraken Trend Bot",
    )
    parser.add_argument("--file", type=str, default=None,
                        help="Path to a specific backtest JSON result")
    parser.add_argument("--all", action="store_true",
                        help="Generate reports for every result file")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run the backtester first, then chart")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║    BACKTEST VISUAL REPORT GENERATOR              ║")
    print("  ╚══════════════════════════════════════════════════╝")

    # Optional: re-run backtester
    if args.rerun:
        print("\n  Re-running backtest ...")
        subprocess.run(
            [sys.executable, "-m", "backtest.backtester"],
            cwd=str(REPO_ROOT),
        )

    # Determine which file(s) to process
    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = RESULTS_DIR / path
        if not path.exists():
            print(f"  ERROR: File not found: {path}")
            return
        generate_report(path)

    elif args.all:
        results = find_all_results()
        if not results:
            print(f"  ERROR: No results in {RESULTS_DIR}")
            return
        print(f"  Found {len(results)} result file(s)\n")
        for p in results:
            generate_report(p)

    else:
        path = find_latest_result()
        if not path:
            print("  ERROR: No backtest results found.")
            print(f"  Run a backtest first:  python -m backtest.backtester")
            return
        generate_report(path)

    print()


if __name__ == "__main__":
    main()
