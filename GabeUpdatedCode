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

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        LIMIT = 80
        PEPPER = "INTARIAN_PEPPER_ROOT"
        OSMIUM = "ASH_COATED_OSMIUM"
        
        # Initialize history if empty
        td = {"pepper_mids": [], "OSMIUM_HISTORY": []}
        if state.traderData:
            try: td = json.loads(state.traderData)
            except: pass

        for product in [PEPPER, OSMIUM]:
            depth: OrderDepth = state.order_depths.get(product)
            if not (depth and depth.buy_orders and depth.sell_orders): continue
                
            orders: List[Order] = []
            pos = state.position.get(product, 0)
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            
            # 1. VWMP Calculation (Improved Noise Filtering)
            ask_vol = abs(depth.sell_orders[best_ask])
            bid_vol = depth.buy_orders[best_bid]
            vw_mid = ((best_bid * ask_vol) + (best_ask * bid_vol)) / (ask_vol + bid_vol)
            
            # --- OSMIUM LOGIC (Mean Reversion + Aggressive Skew) ---
            if product == OSMIUM:
                history = td.get("OSMIUM_HISTORY", [])
                history.append(vw_mid)
                if len(history) > 30: history.pop(0)
                td["OSMIUM_HISTORY"] = history
                
                if len(history) >= 10:
                    sma = sum(history) / len(history)
                    if abs(sma - 10000) > 15: sma = 10000 # Anchor
                    
                    # 2. Exponential Inventory Skew (Robust Risk Mgmt)
                    # We move the thresholds faster as we approach the limit
                    risk_offset = (pos / LIMIT) * 4.0 
                    buy_thr = sma - 4.5 - risk_offset
                    sell_thr = sma + 4.5 - risk_offset

                    # Market Taking
                    for price, vol in sorted(depth.sell_orders.items()):
                        if price <= buy_thr and pos < LIMIT:
                            buy_qty = min(abs(vol), LIMIT - pos)
                            orders.append(Order(product, int(price), int(buy_qty)))
                            pos += buy_qty
                                
                    for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                        if price >= sell_thr and pos > -LIMIT:
                            sell_qty = min(abs(vol), pos + LIMIT)
                            orders.append(Order(product, int(price), int(-sell_qty)))
                            pos -= sell_qty

                    # Market Making (Pennying skewed by inventory)
                    if pos < LIMIT:
                        mm_buy = min(best_bid + 1, int(sma - 2 - risk_offset))
                        orders.append(Order(product, mm_buy, LIMIT - pos))
                    if pos > -LIMIT:
                        mm_sell = max(best_ask - 1, int(sma + 2 - risk_offset))
                        orders.append(Order(product, mm_sell, -LIMIT - pos))

            # --- PEPPER LOGIC (Linear Regression Trend + VWMP) ---
            elif product == PEPPER:
                hist = td.get("pepper_mids", [])
                hist.append(vw_mid)
                if len(hist) > 40: hist.pop(0)
                td["pepper_mids"] = hist

                if len(hist) >= 20:
                    # 3. Linear Regression (Robust Prediction)
                    n = len(hist)
                    x = list(range(n))
                    y = hist
                    sum_x, sum_y = sum(x), sum(y)
                    sum_xx = sum(i*i for i in x)
                    sum_xy = sum(i*j for i, j in zip(x, y))
                    denom = (n * sum_xx - sum_x**2)
                    slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0
                    predicted_fair = (sum_y - slope * sum_x) / n + slope * (n + 1)
                    
                    # Trend Direction
                    direction = 1 if slope > 0.05 else (-1 if slope < -0.05 else 0)

                    # Market Taking skewed by trend
                    if direction == 1:
                        for price, vol in sorted(depth.sell_orders.items()):
                            if price <= predicted_fair + 0.5 and pos < LIMIT:
                                qty = min(abs(vol), LIMIT - pos)
                                orders.append(Order(product, price, qty))
                                pos += qty
                    elif direction == -1:
                        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                            if price >= predicted_fair - 0.5 and pos > -LIMIT:
                                qty = min(abs(vol), pos + LIMIT)
                                orders.append(Order(product, price, -qty))
                                pos -= qty

                    # Strategic Passive Orders (Laddered-style clips)
                    clip_size = 20 # Protect against whipsaws
                    if pos < LIMIT:
                        buy_p = int(min(best_bid + 1, predicted_fair - 1.5))
                        orders.append(Order(product, buy_p, min(LIMIT - pos, clip_size)))
                    if pos > -LIMIT:
                        sell_p = int(max(best_ask - 1, predicted_fair + 1.5))
                        orders.append(Order(product, sell_p, -min(pos + LIMIT, clip_size)))

            result[product] = orders

        td_out = json.dumps(td)
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out
