import json
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data, self.compress_listings(state.listings), self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades), self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [d.buy_orders, d.sell_orders] for s, d in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for t in arr:
                compressed.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        obs = {p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
               for p, o in observations.conversionObservations.items()}
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
        if len(json.dumps(value)) <= max_length: return value
        return value[:max_length-3] + "..."

logger = Logger()

LIMIT = 80
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
OSMIUM_FAIR = 10000

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0

        td = {"pepper_mids": []}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: pass

        # ===== OSMIUM: stable fair-value market making around 10000 =====
        depth = state.order_depths.get(OSMIUM)
        if depth and depth.buy_orders and depth.sell_orders:
            orders: List[Order] = []
            pos = state.position.get(OSMIUM, 0)
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)

            # Take any sell <= FAIR-1 (free edge) and any buy >= FAIR+1
            for price, vol in sorted(depth.sell_orders.items()):
                if price <= OSMIUM_FAIR - 1 and pos < LIMIT:
                    qty = min(-vol, LIMIT - pos)
                    orders.append(Order(OSMIUM, price, qty))
                    pos += qty
            for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                if price >= OSMIUM_FAIR + 1 and pos > -LIMIT:
                    qty = min(vol, pos + LIMIT)
                    orders.append(Order(OSMIUM, price, -qty))
                    pos -= qty

            # Inventory skew on the MM quotes
            skew = -1 if pos > LIMIT * 0.5 else (1 if pos < -LIMIT * 0.5 else 0)
            mm_buy = max(best_bid + 1, OSMIUM_FAIR - 1) + skew
            mm_sell = min(best_ask - 1, OSMIUM_FAIR + 1) + skew
            if mm_buy < mm_sell:
                if pos < LIMIT:
                    orders.append(Order(OSMIUM, mm_buy, LIMIT - pos))
                if pos > -LIMIT:
                    orders.append(Order(OSMIUM, mm_sell, -LIMIT - pos))

            result[OSMIUM] = orders

        # PEPPER trends smoothly (+0.1/call on round-1 data) with a wide spread (~13).
        # Crossing the spread is too costly, so accumulate via passive bids when trend
        # is up and flip sides when slope reverses.
        depth = state.order_depths.get(PEPPER)
        if depth and depth.buy_orders and depth.sell_orders:
            orders = []
            pos = state.position.get(PEPPER, 0)
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid = (best_bid + best_ask) / 2

            hist = td.get("pepper_mids", [])
            hist.append(mid)
            if len(hist) > 100: hist.pop(0)
            td["pepper_mids"] = hist

            slope = 0.0
            if len(hist) >= 30:
                slope = (hist[-1] - hist[0]) / (len(hist) - 1)

            if slope > 0.02:
                direction = 1
            elif slope < -0.02:
                direction = -1
            else:
                direction = 0

            if direction == 0:
                buy_price = best_bid + 1
                sell_price = best_ask - 1
                if buy_price < sell_price:
                    if pos < LIMIT:
                        orders.append(Order(PEPPER, buy_price, min(LIMIT - pos, 20)))
                    if pos > -LIMIT:
                        orders.append(Order(PEPPER, sell_price, -min(LIMIT + pos, 20)))
            elif direction == 1:
                if pos < 0:
                    for price, vol in sorted(depth.sell_orders.items()):
                        if pos >= 0: break
                        qty = min(-vol, -pos)
                        if qty > 0:
                            orders.append(Order(PEPPER, price, qty))
                            pos += qty
                if pos < LIMIT:
                    buy_price = min(best_bid + 1, best_ask - 1)
                    orders.append(Order(PEPPER, buy_price, LIMIT - pos))
                if pos >= LIMIT - 5:
                    sell_price = max(best_ask - 1, int(round(mid)) + 4)
                    orders.append(Order(PEPPER, sell_price, -min(15, LIMIT + pos)))
            else:
                if pos > 0:
                    for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                        if pos <= 0: break
                        qty = min(vol, pos)
                        if qty > 0:
                            orders.append(Order(PEPPER, price, -qty))
                            pos -= qty
                if pos > -LIMIT:
                    sell_price = max(best_ask - 1, best_bid + 1)
                    orders.append(Order(PEPPER, sell_price, -LIMIT - pos))
                if pos <= -(LIMIT - 5):
                    buy_price = min(best_bid + 1, int(round(mid)) - 4)
                    orders.append(Order(PEPPER, buy_price, min(15, LIMIT - pos)))

            result[PEPPER] = orders

        td_out = json.dumps(td)
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
