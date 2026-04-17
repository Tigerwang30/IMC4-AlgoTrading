import json
from typing import Any, List, Dict

from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {s: [d.buy_orders, d.sell_orders] for s, d in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        obs = {
            p: [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
            for p, o in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, obs]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for o in arr:
                compressed.append([o.symbol, o.price, o.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(json.dumps(value)) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    LIMIT = 80

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    # Osmium mean-reversion params
    OSMIUM_HISTORY_LEN = 30
    OSMIUM_MIN_HISTORY = 10
    OSMIUM_ANCHOR_PRICE = 10000
    OSMIUM_ANCHOR_THRESHOLD = 10
    OSMIUM_TAKE_EDGE = 4.5
    OSMIUM_MAKE_EDGE = 2
    OSMIUM_RISK_SCALE = 4.0
    OSMIUM_COOLOFF_TRIGGER = 8
    OSMIUM_COOLOFF_WINDOW = 3
    OSMIUM_COOLOFF_DURATION = 20
    OSMIUM_COOLOFF_EDGE = 7.0

    # Pepper: combine drift capture (positive +0.1/tick drift) with MR takes
    PEPPER_HISTORY_LEN = 40
    PEPPER_MIN_HISTORY = 20
    PEPPER_SLOPE_THRESHOLD = 0.04
    PEPPER_TREND_TAKE_EDGE = 0.5
    PEPPER_MR_TAKE_EDGE = 4.0
    PEPPER_MAKE_EDGE = 1.0
    PEPPER_RISK_SCALE = 1.5
    PEPPER_MM_CLIP = 35
    PEPPER_MM_CLIP_WITH_TREND = 50

    # Shared inventory management
    AGGRESSIVE_CLEAR_FRAC = 0.7

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        td: Dict[str, Any] = {
            "OSMIUM_HISTORY": [],
            "pepper_mids": [],
            "osmium_cooloff": {"ticks_left": 0, "side": None},
            "last_osmium_fills": [],
        }
        if state.traderData:
            try:
                loaded = json.loads(state.traderData)
                if isinstance(loaded, dict):
                    td.update(loaded)
            except Exception:
                pass

        # Update Osmium adverse-fill cool-off based on own_trades since last tick
        self._update_osmium_cooloff(state, td)

        for product in [self.PEPPER, self.OSMIUM]:
            depth: OrderDepth = state.order_depths.get(product)
            if not (depth and depth.buy_orders and depth.sell_orders):
                continue

            orders: List[Order] = []
            pos = state.position.get(product, 0)
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())

            ask_vol = abs(depth.sell_orders[best_ask])
            bid_vol = depth.buy_orders[best_bid]
            vw_mid = ((best_bid * ask_vol) + (best_ask * bid_vol)) / (ask_vol + bid_vol)

            if product == self.OSMIUM:
                orders = self._trade_osmium(product, depth, pos, vw_mid, td)
            elif product == self.PEPPER:
                orders = self._trade_pepper(product, depth, pos, best_bid, best_ask, vw_mid, td)

            result[product] = orders

        td_out = json.dumps(td)
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out

    def _update_osmium_cooloff(self, state: TradingState, td: Dict[str, Any]) -> None:
        cooloff = td.get("osmium_cooloff") or {"ticks_left": 0, "side": None}
        if cooloff.get("ticks_left", 0) > 0:
            cooloff["ticks_left"] -= 1
            if cooloff["ticks_left"] <= 0:
                cooloff = {"ticks_left": 0, "side": None}

        fills_buf = td.get("last_osmium_fills", [])
        own = state.own_trades.get(self.OSMIUM, []) or []
        for t in own:
            # Buy fill: qty>0 when we bought (state.position logic handles sign).
            # Trade objects expose price/quantity; rely on buyer/seller when available.
            try:
                if getattr(t, "buyer", None) == "SUBMISSION":
                    fills_buf.append([state.timestamp, float(t.price), int(t.quantity)])
                elif getattr(t, "seller", None) == "SUBMISSION":
                    fills_buf.append([state.timestamp, float(t.price), -int(t.quantity)])
            except Exception:
                continue

        # Trim buffer to recent entries
        cutoff = state.timestamp - self.OSMIUM_COOLOFF_WINDOW * 100 - 1000
        fills_buf = [f for f in fills_buf if f[0] >= cutoff][-20:]

        depth = state.order_depths.get(self.OSMIUM)
        if depth and depth.buy_orders and depth.sell_orders:
            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
            for ts, px, qty in fills_buf:
                age_ticks = (state.timestamp - ts) / 100
                if 0 <= age_ticks <= self.OSMIUM_COOLOFF_WINDOW:
                    if qty > 0 and mid <= px - self.OSMIUM_COOLOFF_TRIGGER:
                        cooloff = {"ticks_left": self.OSMIUM_COOLOFF_DURATION, "side": "buy"}
                    elif qty < 0 and mid >= px + self.OSMIUM_COOLOFF_TRIGGER:
                        cooloff = {"ticks_left": self.OSMIUM_COOLOFF_DURATION, "side": "sell"}

        td["osmium_cooloff"] = cooloff
        td["last_osmium_fills"] = fills_buf

    def _trade_osmium(
        self,
        product: str,
        depth: OrderDepth,
        pos: int,
        vw_mid: float,
        td: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []

        history = td.get("OSMIUM_HISTORY", [])
        history.append(vw_mid)
        if len(history) > self.OSMIUM_HISTORY_LEN:
            history.pop(0)
        td["OSMIUM_HISTORY"] = history

        if len(history) < self.OSMIUM_MIN_HISTORY:
            return orders

        sma = sum(history) / len(history)
        if abs(sma - self.OSMIUM_ANCHOR_PRICE) > self.OSMIUM_ANCHOR_THRESHOLD:
            sma = self.OSMIUM_ANCHOR_PRICE

        cooloff = td.get("osmium_cooloff") or {"ticks_left": 0, "side": None}
        take_edge = self.OSMIUM_TAKE_EDGE
        cool_side = None
        if cooloff.get("ticks_left", 0) > 0:
            take_edge = self.OSMIUM_COOLOFF_EDGE
            cool_side = cooloff.get("side")

        risk_offset = (pos / self.LIMIT) * self.OSMIUM_RISK_SCALE
        buy_thr = sma - take_edge - risk_offset
        sell_thr = sma + take_edge - risk_offset

        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        for price, vol in sorted(depth.sell_orders.items()):
            if price <= buy_thr and pos < self.LIMIT:
                buy_qty = min(abs(vol), self.LIMIT - pos)
                orders.append(Order(product, int(price), int(buy_qty)))
                pos += buy_qty

        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
            if price >= sell_thr and pos > -self.LIMIT:
                sell_qty = min(abs(vol), pos + self.LIMIT)
                orders.append(Order(product, int(price), int(-sell_qty)))
                pos -= sell_qty

        heavy_long = pos > self.AGGRESSIVE_CLEAR_FRAC * self.LIMIT
        heavy_short = pos < -self.AGGRESSIVE_CLEAR_FRAC * self.LIMIT

        if pos < self.LIMIT and cool_side != "buy" and not heavy_long:
            mm_buy = min(best_bid + 1, int(sma - self.OSMIUM_MAKE_EDGE - risk_offset))
            size = self.LIMIT - pos
            if heavy_short:
                size = min(size, self.LIMIT)
            orders.append(Order(product, mm_buy, size))
        if pos > -self.LIMIT and cool_side != "sell" and not heavy_short:
            mm_sell = max(best_ask - 1, int(sma + self.OSMIUM_MAKE_EDGE - risk_offset))
            size = -self.LIMIT - pos
            orders.append(Order(product, mm_sell, size))

        return orders

    # PEPPER: drift capture via OLS slope + MR takes on big dislocations + clipped MM.
    # Data shows positive drift (+0.1/tick, ~+1000/day) and lag-1 return autocorr ≈ -0.5.
    # The old trend-follow captured drift because slope was positive most of the time;
    # combining that with wide-edge MR takes gives us both sides of the signal.
    def _trade_pepper(
        self,
        product: str,
        depth: OrderDepth,
        pos: int,
        best_bid: int,
        best_ask: int,
        vw_mid: float,
        td: Dict[str, Any],
    ) -> List[Order]:
        orders: List[Order] = []

        hist = td.get("pepper_mids", [])
        hist.append(vw_mid)
        if len(hist) > self.PEPPER_HISTORY_LEN:
            hist.pop(0)
        td["pepper_mids"] = hist

        if len(hist) < self.PEPPER_MIN_HISTORY:
            return orders

        n = len(hist)
        sum_x = n * (n - 1) / 2
        sum_y = sum(hist)
        sum_xx = sum(i * i for i in range(n))
        sum_xy = sum(i * y for i, y in enumerate(hist))
        denom = n * sum_xx - sum_x * sum_x
        slope = (n * sum_xy - sum_x * sum_y) / denom if denom else 0.0
        intercept = (sum_y - slope * sum_x) / n
        predicted = intercept + slope * (n + 1)

        if slope > self.PEPPER_SLOPE_THRESHOLD:
            direction = 1
        elif slope < -self.PEPPER_SLOPE_THRESHOLD:
            direction = -1
        else:
            direction = 0

        # 1. Trend take: drift capture when slope is clearly positive/negative.
        if direction == 1:
            for price, vol in sorted(depth.sell_orders.items()):
                if price <= predicted + self.PEPPER_TREND_TAKE_EDGE and pos < self.LIMIT:
                    qty = min(abs(vol), self.LIMIT - pos)
                    orders.append(Order(product, int(price), int(qty)))
                    pos += qty
        elif direction == -1:
            for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                if price >= predicted - self.PEPPER_TREND_TAKE_EDGE and pos > -self.LIMIT:
                    qty = min(abs(vol), pos + self.LIMIT)
                    orders.append(Order(product, int(price), int(-qty)))
                    pos -= qty

        # 2. Mean-reversion take on extreme dislocations (rarely fires; pure alpha).
        for price, vol in sorted(depth.sell_orders.items()):
            if price <= predicted - self.PEPPER_MR_TAKE_EDGE and pos < self.LIMIT:
                qty = min(abs(vol), self.LIMIT - pos)
                orders.append(Order(product, int(price), int(qty)))
                pos += qty
        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
            if price >= predicted + self.PEPPER_MR_TAKE_EDGE and pos > -self.LIMIT:
                qty = min(abs(vol), pos + self.LIMIT)
                orders.append(Order(product, int(price), int(-qty)))
                pos -= qty

        # 3. Market-making: penny inside book, clipped, with light inventory skew.
        risk_offset = (pos / self.LIMIT) * self.PEPPER_RISK_SCALE
        clip = self.PEPPER_MM_CLIP_WITH_TREND if direction != 0 else self.PEPPER_MM_CLIP

        if pos < self.LIMIT:
            buy_p = int(min(best_bid + 1, predicted - self.PEPPER_MAKE_EDGE - risk_offset))
            orders.append(Order(product, buy_p, min(self.LIMIT - pos, clip)))
        if pos > -self.LIMIT:
            sell_p = int(max(best_ask - 1, predicted + self.PEPPER_MAKE_EDGE - risk_offset))
            orders.append(Order(product, sell_p, -min(pos + self.LIMIT, clip)))

        return orders
