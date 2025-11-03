"""Simple weekly rotation backtest using Qlib.

本示例展示如何使用 Qlib 框架实现“周五收盘买入、下周一开盘卖出”的简单策略，
并在指定的回测区间内对单只股票进行回测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import qlib
from qlib.backtest import Order, backtest_loop, create_account_instance, get_exchange
from qlib.backtest.decision import OrderHelper, TradeDecisionWO
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.constant import REG_CN
from qlib.strategy.base import BaseStrategy
from qlib.tests.data import GetData


@dataclass
class WeeklyStrategyConfig:
    """策略配置。"""

    instrument: str


class WeeklyBuyFridaySellMondayStrategy(BaseStrategy):
    """周五收盘买入、下周一开盘卖出的简单轮动策略。"""

    def __init__(self, config: WeeklyStrategyConfig) -> None:
        super().__init__()
        self.config = config

    def _create_order_helper(self) -> OrderHelper:
        return self.trade_exchange.get_order_helper()

    def _current_position_amount(self) -> float:
        position = self.trade_position
        if position.check_stock(self.config.instrument):
            return position.get_stock_amount(self.config.instrument)
        return 0.0

    def _estimate_buy_amount(
        self,
        cash: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> float:
        """根据可用现金和交易价格估算可买入的数量。"""

        if cash <= 0:
            return 0.0
        deal_price = self.trade_exchange.get_deal_price(
            self.config.instrument,
            start_time,
            end_time,
            direction=Order.BUY,
        )
        if deal_price is None or not np.isfinite(deal_price) or deal_price <= 0:
            return 0.0
        # 预留最小手续费，避免因为费用导致下单失败。
        min_cost = getattr(self.trade_exchange, "min_cost", 0.0) or 0.0
        cash_after_fee = max(cash - min_cost, 0.0)
        if cash_after_fee <= 0:
            return 0.0
        return cash_after_fee / deal_price

    def generate_trade_decision(self, execute_result: List[object] | None = None) -> TradeDecisionWO:
        trade_step = self.trade_calendar.get_trade_step()
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        weekday = start_time.weekday()

        orders = []
        helper = self._create_order_helper()

        tradable = self.trade_exchange.is_stock_tradable(
            stock_id=self.config.instrument,
            start_time=start_time,
            end_time=end_time,
        )
        if not tradable:
            return TradeDecisionWO(orders, self)

        if weekday == 4:  # 周五，使用收盘价买入
            if self._current_position_amount() <= 0:
                cash = self.trade_position.get_cash()
                amount = self._estimate_buy_amount(cash, start_time, end_time)
                if amount > 0:
                    orders.append(
                        helper.create(
                            code=self.config.instrument,
                            amount=amount,
                            direction=Order.BUY,
                            start_time=start_time,
                            end_time=end_time,
                        )
                    )
        elif weekday == 0:  # 周一，使用开盘价卖出
            current_amount = self._current_position_amount()
            if current_amount > 0:
                orders.append(
                    helper.create(
                        code=self.config.instrument,
                        amount=current_amount,
                        direction=Order.SELL,
                        start_time=start_time,
                        end_time=end_time,
                    )
                )

        return TradeDecisionWO(orders, self)


def run_backtest() -> None:
    # 准备数据
    provider_uri = "~/.qlib/qlib_data/cn_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    start_time = "2020-01-01"
    end_time = "2020-02-25"
    benchmark = "SH000300"

    exchange = get_exchange(
        freq="day",
        start_time=start_time,
        end_time=end_time,
        codes=["SH600446"],
        deal_price=("$close", "$open"),
        limit_threshold=0.095,
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
    )

    account = create_account_instance(
        start_time=start_time,
        end_time=end_time,
        benchmark=benchmark,
        account=1_000_000,
    )

    common_infra = CommonInfrastructure(
        trade_exchange=exchange,
        trade_account=account,
    )

    executor = SimulatorExecutor(
        time_per_step="day",
        start_time=start_time,
        end_time=end_time,
        common_infra=common_infra,
        generate_portfolio_metrics=True,
    )

    strategy = WeeklyBuyFridaySellMondayStrategy(
        WeeklyStrategyConfig(instrument="SH600446"),
    )

    portfolio_dict, indicator_dict = backtest_loop(
        start_time=start_time,
        end_time=end_time,
        trade_strategy=strategy,
        trade_executor=executor,
    )

    print("回测收益指标：")
    for freq_key, (indicator_df, _) in indicator_dict.items():
        print(f"频率 {freq_key}：")
        print(indicator_df)

    print("\n组合价值曲线（尾部）：")
    for freq_key, (portfolio_df, _) in portfolio_dict.items():
        print(f"频率 {freq_key}：")
        print(portfolio_df.tail())


if __name__ == "__main__":
    run_backtest()
