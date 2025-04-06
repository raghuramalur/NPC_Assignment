"""this file my_strategy.py should be placed in /scripts folder for this to get 
   recognized and be able to run on hummingbot"""

import logging
import os
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from decimal import Decimal
from typing import Dict, List

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel, ClientFieldData
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

class MyStrategyConfig(BaseClientModel):
    script_file_name: str = Field(default_factory=lambda: os.path.basename(__file__))
    exchange: str = Field("binance", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Exchange to use"))
    trading_pair: str = Field("BTC-USDT", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Trading pair"))
    order_amount: Decimal = Field(0.001, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Order amount"))
    price_type: str = Field("mid", client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Price type to use (mid or last)"))
    refresh_interval: int = Field(30, client_data=ClientFieldData(prompt_on_new=True, prompt=lambda mi: "Refresh time in seconds"))
    candle_interval: str = Field("1m", client_data=ClientFieldData(prompt_on_new=False, prompt=lambda mi: "Candle interval (e.g., 1m, 5m, 1h)"))
    candle_max_records: int = Field(
        default=100, # Using 100 records for indicator stability
        client_data=ClientFieldData(prompt_on_new=False, prompt=lambda mi: "Max candle records to fetch/store"))
    target_inventory_pct: Decimal = Field(
        default=Decimal("0.5"), # Target 50% base asset value
        client_data=ClientFieldData(
            prompt=lambda mi: "Target inventory percentage for base asset (e.g., 0.5 for 50%)",
            prompt_on_new=True)
    )
    inventory_skew_intensity: Decimal = Field(
        default=Decimal("0.5"), # Controls inventory skew sensitivity
        client_data=ClientFieldData(
            prompt=lambda mi: "Intensity factor for inventory skew (higher means more skew)",
            prompt_on_new=True)
    )

class MyStrategy(ScriptStrategyBase):
    price_source = PriceType.MidPrice
    create_timestamp = 0
    candles_feed = None

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MyStrategyConfig):
        super().__init__(connectors)
        self.config = config
        self.market = self.connectors[self.config.exchange]
        self.trading_pair = self.config.trading_pair
        self.last_rsi = 50
        self.last_hma = None
        self.last_macd = None
        self.last_signal = None
        self.time_decay_multiplier = Decimal("1.0")
        self.last_order_time = self.current_timestamp

        # Setup and start the candle feed
        candles_config = CandlesConfig(
            connector=self.config.exchange,
            trading_pair=self.config.trading_pair,
            interval=self.config.candle_interval,
            max_records=self.config.candle_max_records
        )
        self.candles_feed = CandlesFactory.get_candle(candles_config)
        self.candles_feed.start()
        self.logger().info(f"Candle feed for {self.config.trading_pair} ({self.config.candle_interval}) initialized and started.")

    @classmethod
    def init_markets(cls, config: MyStrategyConfig):
        cls.markets = {config.exchange: {config.trading_pair}}
        cls.price_source = PriceType.LastTrade if config.price_type == "last" else PriceType.MidPrice

    def on_tick(self):
        # Initial checks before processing tick
        if self.current_timestamp < self.create_timestamp:
            return
        required_candles = 30 # Need enough candles for MACD(26)
        if not self.candles_feed or not self.candles_feed.ready:
            self.logger().info(f"Candle feed not ready yet for {self.config.trading_pair}...")
            return
        if len(self.candles_feed.candles_df) < required_candles:
            self.logger().info(f"Waiting for more candle data ({len(self.candles_feed.candles_df)}/{required_candles} available)...")
            return

        # Always cancel previous orders before placing new ones in this PMM approach
        self.cancel_all_orders()

        # Calculate all indicators based on candle data
        candles_df = self.candles_feed.candles_df
        close_prices = candles_df["close"].tolist()
        high_prices = candles_df["high"].tolist()
        low_prices = candles_df["low"].tolist()

        ema = self._ema(close_prices, 20)           # EMA for Keltner center line
        atr = self._atr(high_prices, low_prices, close_prices, 10) # ATR for Keltner bands
        upper_band = ema[-1] + (1.5 * atr[-1])      # Keltner Upper Band (M=1.5)
        lower_band = ema[-1] - (1.5 * atr[-1])      # Keltner Lower Band (M=1.5)
        rsi = self._rsi(close_prices, 14)[-1]       # RSI for overbought checks
        hma = self._hma(close_prices, 20)[-1]       # HMA for trend direction
        macd, signal = self._macd(close_prices)[-1] # MACD for trend confirmation/aggression

        self.last_rsi, self.last_hma, self.last_macd, self.last_signal = rsi, hma, macd, signal # Store for status display potentially
        ref_price = self.market.get_price_by_type(self.trading_pair, self.price_source)
        if not ref_price or ref_price <= 0:
             self.logger().warning(f"Invalid reference price ({ref_price}). Skipping tick.")
             return

        # Determine base spread based on Keltner bandwidth % (volatility tiers)
        band_width_value = Decimal(str(upper_band - lower_band))
        bandwidth_percentage = (band_width_value / ref_price) if band_width_value > 0 else Decimal("0.01") # Handle zero/negative width
        # Tiered spread factors
        if bandwidth_percentage < Decimal("0.005"): base_spread_factor = Decimal("0.001")   # Very Low Vol (< 0.5%) -> Tightest spread
        elif bandwidth_percentage < Decimal("0.01"): base_spread_factor = Decimal("0.002")  # Low Vol (< 1.0%) -> Tighter spread
        elif bandwidth_percentage < Decimal("0.02"): base_spread_factor = Decimal("0.005")  # Medium Vol (< 2.0%) -> Wider spread
        else: base_spread_factor = Decimal("0.010")                                         # High Vol (>= 2.0%) -> Widest spread
        spread_factor = base_spread_factor
        self.logger().info(f"Volatility (Bandwidth %): {bandwidth_percentage:.4f}, Base Spread Factor: {spread_factor}")

        # Apply time decay to widen spread if orders are stale
        time_since_last = self.current_timestamp - self.last_order_time
        if time_since_last > self.config.refresh_interval: self.time_decay_multiplier += Decimal("0.05")
        else: self.time_decay_multiplier = max(Decimal("1.0"), self.time_decay_multiplier - Decimal("0.01"))
        spread_factor *= self.time_decay_multiplier
        self.logger().info(f"Spread factor after time decay: {spread_factor}")

        # Apply HMA/MACD trend aggression by adjusting spreads
        buy_spread_factor, sell_spread_factor = spread_factor, spread_factor
        if hma > ref_price and macd > signal: # Bullish trend signal
             buy_spread_factor = spread_factor / Decimal("2"); self.logger().info("HMA/MACD Bullish: Tightening buy spread.")
        elif hma < ref_price and macd < signal: # Bearish trend signal
             sell_spread_factor = spread_factor / Decimal("2"); self.logger().info("HMA/MACD Bearish: Tightening sell spread.")

        # Calculate initial bid/ask prices using adjusted spreads
        initial_bid_price = ref_price * (Decimal("1") - buy_spread_factor)
        initial_ask_price = ref_price * (Decimal("1") + sell_spread_factor)

        # --- Inventory Management Section ---
        allow_buy_inventory, allow_sell_inventory = True, True # Assume allowed by default
        final_bid_price, final_ask_price = initial_bid_price, initial_ask_price # Use initial prices if inventory check fails

        try:
            # Get current balances and inventory percentage
            base_asset, quote_asset = self.trading_pair.split("-")
            base_balance = self.market.get_available_balance(base_asset)
            quote_balance = self.market.get_available_balance(quote_asset)
            if base_balance is None or quote_balance is None: raise ValueError("Balances unavailable.")

            base_value = base_balance * ref_price
            total_portfolio_value = base_value + quote_balance
            current_base_pct = (base_value / total_portfolio_value) if total_portfolio_value > 0 else self.config.target_inventory_pct

            # Define inventory caps based on current volatility level (Keltner bandwidth %)
            # Tighter caps in higher volatility
            if bandwidth_percentage < Decimal("0.01"): # Low Vol
                max_allowed_base_pct = self.config.target_inventory_pct + Decimal("0.10") # +/- 10% from target
                min_allowed_base_pct = self.config.target_inventory_pct - Decimal("0.10")
                vol_level = "Low"
            elif bandwidth_percentage < Decimal("0.02"): # Medium Vol
                max_allowed_base_pct = self.config.target_inventory_pct + Decimal("0.05") # +/- 5% from target
                min_allowed_base_pct = self.config.target_inventory_pct - Decimal("0.05")
                vol_level = "Medium"
            else: # High Vol
                max_allowed_base_pct = self.config.target_inventory_pct # +/- 0% from target (strict)
                min_allowed_base_pct = self.config.target_inventory_pct
                vol_level = "High"
            max_allowed_base_pct = min(Decimal("1.0"), max(Decimal("0.0"), max_allowed_base_pct)) # Ensure caps are valid [0,1]
            min_allowed_base_pct = min(Decimal("1.0"), max(Decimal("0.0"), min_allowed_base_pct)) # Ensure caps are valid [0,1]

            # Check inventory caps and block orders if necessary
            if current_base_pct >= max_allowed_base_pct:
                allow_buy_inventory = False
                self.logger().info(f"Inv Cap ({vol_level}): Base {current_base_pct*100:.1f}% >= Max {max_allowed_base_pct*100:.1f}%. BUYS BLOCKED.")
            if current_base_pct <= min_allowed_base_pct:
                allow_sell_inventory = False
                self.logger().info(f"Inv Cap ({vol_level}): Base {current_base_pct*100:.1f}% <= Min {min_allowed_base_pct*100:.1f}%. SELLS BLOCKED.")

            # Calculate inventory skew adjustment
            inventory_deviation = current_base_pct - self.config.target_inventory_pct
            price_skew = ref_price * inventory_deviation * self.config.inventory_skew_intensity
            skewed_bid_price = initial_bid_price - price_skew # Shift midpoint based on inventory deviation
            skewed_ask_price = initial_ask_price - price_skew

            self.logger().info(
                f"Inventory: {current_base_pct*100:.1f}% Base ({base_balance} {base_asset}), "
                f"Target: {self.config.target_inventory_pct*100:.1f}%, "
                f"Deviation: {inventory_deviation*100:.1f}%, "
                f"Price Skew: {price_skew:.4f}. "
                f"Final Prices: Bid={skewed_bid_price:.4f}, Ask={skewed_ask_price:.4f}"
            )

            # Set final prices to skewed prices
            final_bid_price = skewed_bid_price
            final_ask_price = skewed_ask_price

        except Exception as e:
            self.logger().error(f"Error during inventory management: {str(e)}. Using initial prices/allow flags.")

        # --- End Inventory Management ---


        # Determine final order permissions based on inventory caps and RSI check
        final_allow_buy = allow_buy_inventory # Buys only limited by inventory cap

        rsi_overbought = ref_price > upper_band and rsi > 70 # Check RSI condition
        if rsi_overbought: self.logger().info("RSI Overbought: Condition met.")

        final_allow_sell = allow_sell_inventory and not rsi_overbought # Sells limited by inventory cap OR RSI
        if not allow_sell_inventory: self.logger().info("Sells blocked by inventory cap.")
        elif rsi_overbought: self.logger().info("Sells blocked by RSI overbought condition.")


        # Create order candidates if allowed, using final prices and quantized values
        orders_to_place = []
        if final_allow_buy:
            quantized_buy_price = self.market.quantize_order_price(self.trading_pair, final_bid_price)
            quantized_buy_amount = self.market.quantize_order_amount(self.trading_pair, self.config.order_amount)
            if quantized_buy_amount > 0: # Check non-zero amount after quantization
                 buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT, order_side=TradeType.BUY, amount=quantized_buy_amount, price=quantized_buy_price)
                 orders_to_place.append(buy_order)
            else: self.logger().warning(f"Buy order amount quantized to zero. Original: {self.config.order_amount}")
        else: self.logger().info("Buy order placement skipped.")

        if final_allow_sell:
            quantized_sell_price = self.market.quantize_order_price(self.trading_pair, final_ask_price)
            quantized_sell_amount = self.market.quantize_order_amount(self.trading_pair, self.config.order_amount)
            if quantized_sell_amount > 0: # Check non-zero amount after quantization
                 sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT, order_side=TradeType.SELL, amount=quantized_sell_amount, price=quantized_sell_price)
                 orders_to_place.append(sell_order)
            else: self.logger().warning(f"Sell order amount quantized to zero. Original: {self.config.order_amount}")
        else: self.logger().info("Sell order placement skipped.")

        # Adjust orders for budget and place them
        adjusted_orders = self.market.budget_checker.adjust_candidates(orders_to_place, all_or_none=False) # Allow placing one side if possible
        for order in adjusted_orders:
            if order.price > 0: # Check price validity before placing
                self.place_order(self.config.exchange, order)
            else: self.logger().warning(f"Skipping order placement due to invalid price: {order.price}")

        # Update timestamps for next tick
        self.create_timestamp = self.current_timestamp + self.config.refresh_interval
        self.last_order_time = self.current_timestamp # Update time for decay calculation

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name, order.trading_pair, order.amount, order.order_type, order.price)
        else:
            self.buy(connector_name, order.trading_pair, order.amount, order.order_type, order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = f"{event.trade_type.name} {round(event.amount, 6)} {event.trading_pair} at {round(event.price, 2)}"
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    # --- Indicator Implementations ---

    def _ema(self, data: List[float], period: int) -> List[float]:
        ema = []
        k = 2 / (period + 1)
        for i in range(len(data)):
            if i < period: ema.append(sum(data[:i+1]) / len(data[:i+1]))
            else: ema.append(data[i] * k + ema[-1] * (1 - k))
        return ema

    def _atr(self, high: List[float], low: List[float], close: List[float], period: int) -> List[float]:
        tr = []
        for i in range(1, len(close)):
            tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            tr.append(tr_val)
        atr = []
        for i in range(len(tr)):
            if i < period: atr.append(sum(tr[:i+1]) / len(tr[:i+1]))
            else: atr.append((atr[-1] * (period - 1) + tr[i]) / period)
        return [0] + atr # Pad initial value

    def _rsi(self, data: List[float], period: int) -> List[float]:
        gains, losses, rsi = [], [], []
        for i in range(1, len(data)):
            delta = data[i] - data[i - 1]
            gains.append(max(0, delta))
            losses.append(max(0, -delta))

        if len(gains) >= period:
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0 # Avoid division by zero
            rsi_val = 100 - (100 / (1 + rs))
            rsi.append(rsi_val)
            for i in range(period, len(gains)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi_val = 100 - (100 / (1 + rs))
                rsi.append(rsi_val)

        padding_len = len(data) - len(rsi)
        return [50.0] * padding_len + rsi # Pad with neutral 50

    def _hma(self, data: List[float], period: int) -> List[float]:
        def wma(values, n):
            res = []
            for i in range(len(values)):
                if i + 1 < n: res.append(sum(values[:i+1]) / (i+1)) # SMA for initial points
                else:
                    weights = list(range(1, n + 1))
                    subset = values[i - n + 1:i + 1]
                    res.append(sum(w * v for w, v in zip(weights, subset)) / sum(weights))
            return res
        if period <= 1: return data[:]
        half_length = max(1, period // 2)
        sqrt_length = max(1, int(period**0.5))
        wma_half = wma(data, half_length)
        wma_full = wma(data, period)
        min_len = min(len(wma_half), len(wma_full))
        diff = [(2 * h) - f for h, f in zip(wma_half[-min_len:], wma_full[-min_len:])] # Raw HMA value
        if not diff: return data[:] # Handle empty diff
        hma_final = wma(diff, sqrt_length) # Smoothed HMA
        padding_len = len(data) - len(hma_final)
        padding_value = hma_final[0] if hma_final else (data[0] if data else 0)
        return [padding_value] * padding_len + hma_final

    def _macd(self, data: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> List[tuple]:
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        if not ema_fast or not ema_slow: return [(0.0, 0.0)] * len(data)
        min_len_ema = min(len(ema_fast), len(ema_slow))
        macd_line = [f - s for f, s in zip(ema_fast[-min_len_ema:], ema_slow[-min_len_ema:])] # MACD line calculation
        if not macd_line: return [(0.0, 0.0)] * len(data)
        signal_line = self._ema(macd_line, signal_period) # Signal line calculation
        if not signal_line: return [(0.0, 0.0)] * len(data)
        min_len_macd = min(len(macd_line), len(signal_line))
        macd_tuples = list(zip(macd_line[-min_len_macd:], signal_line[-min_len_macd:])) # Combine MACD and Signal
        padding_len = len(data) - len(macd_tuples)
        padding_value = (0.0, 0.0) # Pad with zero tuple
        return [padding_value] * padding_len + macd_tuples
