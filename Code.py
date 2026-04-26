# =============================================================================
# ROUND 3 - "Gloves Off" - Solvenar
# -----------------------------------------------------------------------------
# MANUAL CHALLENGE - Ornamental Bio-Pods (submit in GUI):
#   Bid 1: 790
#   Bid 2: 870
# Reasoning: reserves uniform on {670,675,...,920} (51 levels), resell at 920.
#   b1: argmax (b-665)*(920-b)/255 -> b = 792.5 -> use 790 (E ~ 63.7 / counterparty)
#   b2: residual range {795..920}; ~857 ignoring penalty, use 870 to stay
#        comfortably above an assumed avg_b2 ~ 855-870 and dampen the cubic penalty.
# =============================================================================

import json
import math
from typing import Any, List, Dict, Optional, Tuple
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


# ---------------- Product constants ----------------
HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

# Vouchers used by the smile model. We deliberately exclude 5200, 5300, 5400 from
# BOTH fit and trade: the 443435 backtest showed they have persistent one-sided
# biases (991-1000 ticks on the same side of theo), which is the fingerprint of a
# smile-fit artifact rather than real alpha. Trading them lost -509 net while
# pulling the curve away from the clean strikes. Skip 4000 (~intrinsic) and
# 6000/6500 (pinned at 0.5) as before.
SMILE_STRIKES: List[int] = [4500, 5000, 5100, 5500]
TRADE_STRIKES: List[int] = [4500, 5000, 5100, 5500]
VOUCHER_SYMS: Dict[int, str] = {k: f"VEV_{k}" for k in SMILE_STRIKES}
ALL_VOUCHER_SYMS: List[str] = [f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]]

# Position limits
LIMITS: Dict[str, int] = {HYDROGEL: 200, VELVET: 200}
for s in ALL_VOUCHER_SYMS:
    LIMITS[s] = 300

VOUCHER_SOFT_LIMIT = 60  # 4 clean strikes only - reduced concentration risk
HYDROGEL_FAIR = 10000

# Time-to-expiry: Round 3 starts at TTE = 5 days; one round = 1_000_000 timestamp units
TTE_START_DAYS = 5.0
ROUND_LEN = 1_000_000


# ---------------- Black-Scholes helpers ----------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or T <= 0:
        return None
    lo, hi = 1e-4, 3.0
    if bs_call_price(S, K, T, hi) < price:
        return None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fit_quadratic_smile(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """Closed-form least squares for iv = a + b*m + c*m^2."""
    if len(points) < 4:
        return None
    n = len(points)
    sx = sx2 = sx3 = sx4 = sy = sxy = sx2y = 0.0
    for m, iv in points:
        sx += m
        sx2 += m * m
        sx3 += m ** 3
        sx4 += m ** 4
        sy += iv
        sxy += m * iv
        sx2y += m * m * iv
    # Normal equations matrix:
    # | n   sx   sx2 | |a|   | sy   |
    # | sx  sx2  sx3 | |b| = | sxy  |
    # | sx2 sx3  sx4 | |c|   | sx2y |
    M = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    Y = [sy, sxy, sx2y]
    try:
        # Cramer's rule via 3x3 determinants
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
                    - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
                    + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
        D = det3(M)
        if abs(D) < 1e-12:
            return None
        Ma = [[Y[i] if j == 0 else M[i][j] for j in range(3)] for i in range(3)]
        Mb = [[Y[i] if j == 1 else M[i][j] for j in range(3)] for i in range(3)]
        Mc = [[Y[i] if j == 2 else M[i][j] for j in range(3)] for i in range(3)]
        return (det3(Ma) / D, det3(Mb) / D, det3(Mc) / D)
    except Exception:
        return None


def vw_mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    bvol = depth.buy_orders[best_bid]
    avol = abs(depth.sell_orders[best_ask])
    if bvol + avol == 0:
        return (best_bid + best_ask) / 2.0
    # Volume-weighted: heavier side pulls mid toward the other side
    return (best_bid * avol + best_ask * bvol) / (avol + bvol)


# ---------------- Trader ----------------
class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        td: Dict[str, Any] = {"hydro_mids": [], "velvet_mids": []}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                pass
        td.setdefault("hydro_mids", [])
        td.setdefault("velvet_mids", [])

        # ===== HYDROGEL_PACK =====
        self._trade_hydrogel(state, td, result)

        # ===== VELVETFRUIT_EXTRACT (also computes spot for vouchers) =====
        velvet_spot = self._trade_velvet(state, td, result)

        # ===== VEV vouchers + delta hedge =====
        if velvet_spot is not None:
            self._trade_vouchers(state, td, result, velvet_spot)

        td_out = json.dumps(td, separators=(",", ":"))
        logger.flush(state, result, conversions, td_out)
        return result, conversions, td_out

    # ---------- HYDROGEL ----------
    def _trade_hydrogel(self, state: TradingState, td: Dict[str, Any], result: Dict[str, List[Order]]) -> None:
        depth = state.order_depths.get(HYDROGEL)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return
        m = vw_mid(depth)
        if m is None:
            return
        hist = td["hydro_mids"]
        hist.append(m)
        if len(hist) > 50:
            hist.pop(0)

        sma = sum(hist) / len(hist)
        # 95% SMA, 5% legacy 10000 anchor (mostly dynamic). Clamp to mid +- 30 so
        # we never quote crazily far from the touch but still follow drift.
        fair = 0.95 * sma + 0.05 * HYDROGEL_FAIR
        fair = max(min(fair, m + 30), m - 30)

        pos = state.position.get(HYDROGEL, 0)
        limit = LIMITS[HYDROGEL]
        orders: List[Order] = []
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())

        # Take any clear mispricing (>= 2.5 inside fair) - tightened from 2 to chase less
        for price, vol in sorted(depth.sell_orders.items()):
            if price <= fair - 2.5 and pos < limit:
                qty = min(-vol, limit - pos)
                if qty > 0:
                    orders.append(Order(HYDROGEL, price, qty))
                    pos += qty
        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
            if price >= fair + 2.5 and pos > -limit:
                qty = min(vol, pos + limit)
                if qty > 0:
                    orders.append(Order(HYDROGEL, price, -qty))
                    pos -= qty

        # Quote ±3 around fair, with stronger inventory skew (was /40) for faster
        # mean-revert at limits - reduces drawdown from one-sided inventory.
        skew = -round(pos / 25.0)
        mm_buy = int(min(best_bid + 1, fair - 3 + skew))
        mm_sell = int(max(best_ask - 1, fair + 3 + skew))
        if mm_buy < mm_sell:
            if pos < limit:
                orders.append(Order(HYDROGEL, mm_buy, limit - pos))
            if pos > -limit:
                orders.append(Order(HYDROGEL, mm_sell, -limit - pos))

        if orders:
            result[HYDROGEL] = orders

    # ---------- VELVET ----------
    def _trade_velvet(self, state: TradingState, td: Dict[str, Any], result: Dict[str, List[Order]]) -> Optional[float]:
        depth = state.order_depths.get(VELVET)
        if not depth or not depth.buy_orders or not depth.sell_orders:
            return None
        m = vw_mid(depth)
        if m is None:
            return None
        hist = td["velvet_mids"]
        hist.append(m)
        if len(hist) > 100:
            hist.pop(0)

        pos = state.position.get(VELVET, 0)
        limit = LIMITS[VELVET]
        orders: List[Order] = []
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid

        # Take aggressive prints relative to vw_mid
        for price, vol in sorted(depth.sell_orders.items()):
            if price <= m - 2 and pos < limit:
                qty = min(-vol, limit - pos)
                if qty > 0:
                    orders.append(Order(VELVET, price, qty))
                    pos += qty
        for price, vol in sorted(depth.buy_orders.items(), reverse=True):
            if price >= m + 2 and pos > -limit:
                qty = min(vol, pos + limit)
                if qty > 0:
                    orders.append(Order(VELVET, price, -qty))
                    pos -= qty

        # Passive quotes one inside the touch (skip if spread too tight)
        if spread >= 2:
            skew = -round(pos / 50.0)
            mm_buy = best_bid + 1 + skew
            mm_sell = best_ask - 1 + skew
            if mm_buy < mm_sell:
                clip = 40  # bumped from 30 since hedging is now passive/rare
                if pos < limit:
                    orders.append(Order(VELVET, int(mm_buy), min(limit - pos, clip)))
                if pos > -limit:
                    orders.append(Order(VELVET, int(mm_sell), -min(limit + pos, clip)))

        if orders:
            result[VELVET] = orders

        return m

    # ---------- VOUCHERS ----------
    def _trade_vouchers(
        self, state: TradingState, td: Dict[str, Any],
        result: Dict[str, List[Order]], spot: float
    ) -> None:
        # Time to expiry in "days" units (vol absorbs the unit scaling)
        T = TTE_START_DAYS - state.timestamp / ROUND_LEN
        if T <= 1e-4:
            return  # nothing useful left

        # 1) per-strike implied vol from the current mid (use SMILE_STRIKES so the
        # excluded TRADE_STRIKES still contribute data points to the fit)
        iv_points: List[Tuple[float, float, int, float]] = []  # (m, iv, K, mid)
        voucher_state: Dict[int, Dict[str, float]] = {}
        for K in SMILE_STRIKES:
            sym = VOUCHER_SYMS[K]
            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0
            iv = implied_vol(mid, spot, K, T)
            voucher_state[K] = {
                "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
                "spread": best_ask - best_bid,
            }
            if iv is None:
                continue
            m = math.log(K / spot)
            iv_points.append((m, iv, K, mid))

        # 2) fit smile, then EMA-smooth with stored coefs to dampen tick-noise
        raw = fit_quadratic_smile([(m, iv) for (m, iv, _, _) in iv_points])
        prev = td.get("smile_coefs")
        if raw is not None:
            if prev is not None and len(prev) == 3:
                alpha = 0.2
                coefs = (
                    alpha * raw[0] + (1 - alpha) * prev[0],
                    alpha * raw[1] + (1 - alpha) * prev[1],
                    alpha * raw[2] + (1 - alpha) * prev[2],
                )
            else:
                coefs = raw
            td["smile_coefs"] = list(coefs)
        elif prev is not None and len(prev) == 3:
            coefs = (prev[0], prev[1], prev[2])  # reuse last good fit
        else:
            coefs = None

        # Track total option delta from desired post-trade positions
        net_option_delta = 0.0

        # 3) generate per-voucher orders ONLY for TRADE_STRIKES
        for K in TRADE_STRIKES:
            sym = VOUCHER_SYMS[K]
            if K not in voucher_state:
                continue
            vs = voucher_state[K]
            depth = state.order_depths[sym]
            pos = state.position.get(sym, 0)
            soft_limit = VOUCHER_SOFT_LIMIT
            mid = vs["mid"]
            spread = vs["spread"]
            best_bid = vs["best_bid"]
            best_ask = vs["best_ask"]
            m_lm = math.log(K / spot)

            if coefs is not None:
                a, b, c = coefs
                fit_iv = a + b * m_lm + c * m_lm * m_lm
                if fit_iv <= 0:
                    fit_iv = 1e-3
                theo = bs_call_price(spot, K, T, fit_iv)
                delta_K = bs_call_delta(spot, K, T, fit_iv)
                # Vega-aware floor: 0.05% IV deviation, scaled to 30% (was 0.5% at
                # 100% which produced 11-45 dollar floors and blocked every trade).
                vega_step = (
                    bs_call_price(spot, K, T, fit_iv + 0.0005)
                    - bs_call_price(spot, K, T, fit_iv - 0.0005)
                )
                iv_noise_thresh = 0.3 * abs(vega_step)
            else:
                theo = mid
                delta_K = 1.0 if spot > K else 0.0
                iv_noise_thresh = 0.0

            half_spread = 0.5 * spread
            threshold = max(half_spread + 0.3, iv_noise_thresh, 0.5)
            clip = 20

            orders: List[Order] = []

            # Strong-edge entry: BUY when offered well below theo
            if mid < theo - threshold and pos < soft_limit:
                for price, vol in sorted(depth.sell_orders.items()):
                    if price > theo - threshold:
                        break
                    if pos >= soft_limit:
                        break
                    qty = min(-vol, soft_limit - pos, clip)
                    if qty > 0:
                        orders.append(Order(sym, price, qty))
                        pos += qty

            # Strong-edge entry: SELL when bid well above theo
            elif mid > theo + threshold and pos > -soft_limit:
                for price, vol in sorted(depth.buy_orders.items(), reverse=True):
                    if price < theo + threshold:
                        break
                    if pos <= -soft_limit:
                        break
                    qty = min(vol, pos + soft_limit, clip)
                    if qty > 0:
                        orders.append(Order(sym, price, -qty))
                        pos -= qty

            else:
                # Mean-revert toward zero when no clear edge
                if abs(mid - theo) < 0.5 * threshold and pos != 0:
                    if pos > 0:
                        scale = min(pos, 15)
                        orders.append(Order(sym, int(best_ask - 1), -scale))
                    else:
                        scale = min(-pos, 15)
                        orders.append(Order(sym, int(best_bid + 1), scale))

            if orders:
                result[sym] = orders

            net_option_delta += pos * delta_K

        # 4) PASSIVE delta hedge with VELVETFRUIT_EXTRACT (only when significant)
        velvet_pos = state.position.get(VELVET, 0)
        pending_velvet = sum(o.quantity for o in result.get(VELVET, []))
        projected_velvet = velvet_pos + pending_velvet
        total_delta = projected_velvet + net_option_delta
        velvet_limit = LIMITS[VELVET]

        if abs(total_delta) >= 25:
            depth_v = state.order_depths.get(VELVET)
            if depth_v and depth_v.buy_orders and depth_v.sell_orders:
                best_bid_v = max(depth_v.buy_orders.keys())
                best_ask_v = min(depth_v.sell_orders.keys())
                hedge_qty = -int(round(total_delta))
                if hedge_qty > 0:
                    capacity = velvet_limit - projected_velvet
                    hedge_qty = min(hedge_qty, max(0, capacity), 30)
                    if hedge_qty > 0:
                        # Passive: bid one tick inside our side
                        price = best_bid_v + 1
                        if price >= best_ask_v:  # don't cross
                            price = best_ask_v - 1
                        result.setdefault(VELVET, []).append(Order(VELVET, price, hedge_qty))
                else:
                    capacity = velvet_limit + projected_velvet
                    hedge_qty = -min(-hedge_qty, max(0, capacity), 30)
                    if hedge_qty < 0:
                        price = best_ask_v - 1
                        if price <= best_bid_v:
                            price = best_bid_v + 1
                        result.setdefault(VELVET, []).append(Order(VELVET, price, hedge_qty))
