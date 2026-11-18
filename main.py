"""Regime-aware cbBTC/USDC trading bot for make.fun bars-replay/v1."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
MIN_NOTIONAL_RAW = 1_000_000  # 1 USDC


class Strategy:
    """Trend-following with ER chop filter, multi-horizon momentum confirmation, and ATR trailing stop."""

    def __init__(self, state):
        spec = state.symbols[SYMBOL]
        self.price_scale = 10 ** int(spec["price_scale"])
        self.qty_scale = 10 ** int(spec["qty_scale"])

        self.closes = []
        self.ema_slow_hist = []

        self.ema_fast = None
        self.ema_med = None
        self.ema_slow = None
        self.atr_bps = 0.0

        self.in_pos = False
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.entry_bar = -1
        self.last_exit_bar = -100
        self.bar_idx = -1

    def _update_ema(self, prev, val, span):
        alpha = 2.0 / (span + 1.0)
        return val if prev is None else alpha * val + (1.0 - alpha) * prev

    def _efficiency_ratio(self, period):
        if len(self.closes) <= period:
            return 0.0
        net = abs(self.closes[-1] - self.closes[-1 - period])
        path = sum(
            abs(self.closes[i] - self.closes[i - 1])
            for i in range(len(self.closes) - period, len(self.closes))
        )
        return net / path if path > 0.0 else 0.0

    def on_batch(self, state, batch):
        bar = next((b for b in batch.get("bars", []) if b["symbol"] == SYMBOL), None)
        if bar is None:
            return []

        self.bar_idx += 1
        close = int(bar["c"]) / self.price_scale
        high = int(bar["h"]) / self.price_scale
        low = int(bar["l"]) / self.price_scale

        prev_close = self.closes[-1] if self.closes else close
        self.closes.append(close)

        tr_bps = ((max(high, prev_close) - min(low, prev_close)) / prev_close) * 10000.0
        self.atr_bps = tr_bps if self.bar_idx == 0 else 0.95 * self.atr_bps + 0.05 * tr_bps

        self.ema_fast = self._update_ema(self.ema_fast, close, 12)
        self.ema_med = self._update_ema(self.ema_med, close, 36)
        self.ema_slow = self._update_ema(self.ema_slow, close, 72)
        self.ema_slow_hist.append(self.ema_slow)

        if state.warming_up or len(self.closes) < 72:
            return []

        pos = state.positions.get(SYMBOL)
        held_qty = pos.qty if pos else 0

        if held_qty > 0:
            if not self.in_pos:
                self.in_pos = True
                self.entry_price = close
                self.peak_price = close
                self.entry_bar = self.bar_idx
            elif close > self.peak_price:
                self.peak_price = close

            hold_bars = self.bar_idx - self.entry_bar
            drawdown_from_peak_bps = ((self.peak_price - close) / self.peak_price) * 10000.0
            pnl_bps = ((close - self.entry_price) / self.entry_price) * 10000.0
            peak_pnl_bps = ((self.peak_price - self.entry_price) / self.entry_price) * 10000.0

            if peak_pnl_bps >= 60.0:
                stop_bps = max(35.0, min(70.0, 2.0 * self.atr_bps))
            else:
                stop_bps = max(50.0, min(100.0, 3.0 * self.atr_bps))

            trend_broken = (self.ema_fast < self.ema_med * 0.9992) and hold_bars >= 10
            hard_stop = pnl_bps < -60.0
            trailing_hit = drawdown_from_peak_bps > stop_bps and hold_bars >= 5

            if hard_stop or trailing_hit or trend_broken:
                self.in_pos = False
                self.last_exit_bar = self.bar_idx
                return [{
                    "side": "sell",
                    "symbol": SYMBOL,
                    "qty": str(held_qty),
                    "max_slippage_bps": 150,
                }]
            return []

        self.in_pos = False
        if self.bar_idx - self.last_exit_bar < 20:
            return []

        er_30 = self._efficiency_ratio(30)
        ret_60_bps = ((close - self.closes[-61]) / self.closes[-61]) * 10000.0
        slow_rising = self.ema_slow > self.ema_slow_hist[-11]
        trend_up = (
            slow_rising
            and ret_60_bps > 25.0
            and self.ema_fast > self.ema_med * 1.0005
            and self.ema_med > self.ema_slow
            and close > self.ema_med
        )
        dist_med_bps = ((close - self.ema_med) / self.ema_med) * 10000.0

        if trend_up and er_30 > 0.32 and 0.0 < dist_med_bps < 55.0:
            depths = state.depth_at(SYMBOL)
            depth = min(depths) if depths else 0
            max_by_impact = (depth * 6) // 10000 if depth > 0 else state.cash
            budget = min(int(state.cash * 0.98), state.weight_room(SYMBOL), max_by_impact)
            if budget >= MIN_NOTIONAL_RAW:
                impact = state.impact_bps(SYMBOL, budget) or 0
                slippage_cap = min(200, max(50, impact + 40))
                return [{
                    "side": "buy",
                    "symbol": SYMBOL,
                    "notional_quote_units": str(budget),
                    "max_slippage_bps": slippage_cap,
                }]

        return []


def main():
    holder = {}

    def on_hello(state):
        holder["strat"] = Strategy(state)

    def on_batch(state, batch):
        return holder["strat"].on_batch(state, batch)

    makefun_bars.run(on_batch, on_hello=on_hello)


if __name__ == "__main__":
    main()
