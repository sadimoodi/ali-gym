from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from ta import add_all_ta_features
import copy
from datetime import datetime
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
import numpy as np
from scipy.special import expit

import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_colors
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import gym
from gym import spaces
from gym.utils import seeding

from ..simulator import MtSimulator, OrderType


class MtEnv(gym.Env):

    metadata = {'render.modes': ['human', 'simple_figure', 'advanced_figure']}

    def __init__(
            self, original_simulator: MtSimulator, trading_symbols: List[str],
            window_size: int, time_points: Optional[List[datetime]]=None,
            hold_threshold: float=0.5, close_threshold: float=0.5,
            #fee: Union[float, Callable[[str], float]]=0.0005,
            symbol_max_orders: int=1, multiprocessing_processes: Optional[int]=None
        ) -> None:

        # validations
        assert len(original_simulator.symbols_data) > 0, "no data available"
        assert len(original_simulator.symbols_info) > 0, "no data available"
        assert len(trading_symbols) > 0, "no trading symbols provided"
        assert 0. <= hold_threshold <= 1., "'hold_threshold' must be in range [0., 1.]"

        if not original_simulator.hedge:
            symbol_max_orders = 1

        for symbol in trading_symbols:
            assert symbol in original_simulator.symbols_info, f"symbol '{symbol}' not found"
            currency_profit = original_simulator.symbols_info[symbol].currency_profit
            assert original_simulator._get_unit_symbol_info(currency_profit) is not None, \
                   f"unit symbol for '{currency_profit}' not found"

        if time_points is None:
            time_points = original_simulator.symbols_data[trading_symbols[0]].index.to_pydatetime().tolist()
        assert len(time_points) > window_size, "not enough time points provided"

        # attributes
        self.seed()
        self.original_simulator = original_simulator
        self.trading_symbols = trading_symbols
        self.window_size = window_size
        self.time_points = time_points
        self.hold_threshold = hold_threshold
        self.close_threshold = close_threshold
        #self.fee = fee
        self.symbol_max_orders = symbol_max_orders
        self.multiprocessing_pool = Pool(multiprocessing_processes) if multiprocessing_processes else None

        self.prices = self._get_prices()
        self.signal_features = self._process_obs()
        self.features_shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.trading_symbols) * (self.symbol_max_orders + 3),)
        )  # symbol -> [close_order_i(logit), hold(logit), volume]

        self.observation_space = spaces.Dict({
            'balance': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'PnL': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'net_worth': spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=self.features_shape),
            'orders': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.trading_symbols), self.symbol_max_orders, 3)
            )  # symbol, order_i -> [entry_price, volume, profit]
        })

        # episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.time_points) - 1
        self._done: bool = NotImplemented
        self._current_tick: int = NotImplemented
        self.simulator: MtSimulator = NotImplemented
        self.history: List[Dict[str, Any]] = NotImplemented


    def seed(self, seed: Optional[int]=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self) -> Dict[str, np.ndarray]:
        self._done = False
        self._current_tick = self._start_tick
        self.simulator = copy.deepcopy(self.original_simulator)
        self.simulator.current_time = self.time_points[self._current_tick]
        self.history = [self._create_info()]
        return self._get_observation()


    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        orders_info, closed_orders_info = self._apply_action(action)
        #print (orders_info)

        self._current_tick += 1
        if self._current_tick == self._end_tick or self.simulator.balance == 0:
            self._done = True

        dt = self.time_points[self._current_tick] - self.time_points[self._current_tick - 1]
        self.simulator.tick(dt)

        step_reward = self._calculate_reward()

        info = self._create_info(
            orders=orders_info, closed_orders=closed_orders_info, step_reward=step_reward
        )
        observation = self._get_observation()
        self.history.append(info)

        return observation, step_reward, self._done, info


    def _apply_action(self, action: np.ndarray) -> Tuple[Dict, Dict]:
        #print (action)
        orders_info = {}
        closed_orders_info = {symbol: [] for symbol in self.trading_symbols}
        
        k = self.symbol_max_orders + 3
        #print (self.trading_symbols)
        for i, symbol in enumerate(self.trading_symbols):
            symbol_action = action[k*i:k*(i+1)]
            close_orders_logit = symbol_action[:-3]
            hold_logit = symbol_action[-3]
            amount = symbol_action[-2]
            leverage = symbol_action[-1]

            close_orders_probability = expit(close_orders_logit)
            hold_probability = expit(hold_logit)
            hold = bool(hold_probability > self.hold_threshold)
            modified_amount = self._get_modified_amount(symbol, amount)
            #print (f'amount= {amount}, modified amount=%.2f' % modified_amount)
            modified_leverage = self._get_modified_leverage(symbol, leverage)
            #print (f'Leverage= {leverage}, modified Lev= {modified_leverage}')

            symbol_orders = self.simulator.symbol_orders(symbol)
            orders_to_close_index = np.where(
                close_orders_probability[:len(symbol_orders)] > self.close_threshold
            )[0]
            orders_to_close = np.array(symbol_orders)[orders_to_close_index]

            for j, order in enumerate(orders_to_close):
                self.simulator.close_order(order)
                closed_orders_info[symbol].append(dict(
                    order_id=order.id, symbol=order.symbol, order_type=order.type,
                    amount=order.amount, volume= order.volume,fee=order.fee, profit=order.profit,
                    close_probability=close_orders_probability[orders_to_close_index][j],
                    leverage=order.leverage,
                ))

            orders_capacity = self.symbol_max_orders - (len(symbol_orders) - len(orders_to_close))
            orders_info[symbol] = dict(
                order_id=None, symbol=symbol, hold_probability=hold_probability,
                hold=hold, amount= modified_amount, capacity=orders_capacity,
                order_type=OrderType.Buy if amount > 0. else OrderType.Sell,
                volume = float('nan'), leverage=None, fee=float('nan'),error='', 
            )

            if self.simulator.hedge and orders_capacity == 0:
                orders_info[symbol].update(dict(
                    error="cannot add more orders"
                ))
            elif modified_amount < 10:
                orders_info[symbol].update(dict(
                    error=f"Amount: {modified_amount:.2f} < 10"
                ))
            elif not hold:
                order_type = OrderType.Buy if amount > 0. else OrderType.Sell
                #fee = self.fee if type(self.fee) is float else self.fee(symbol)

                try:
                    order = self.simulator.create_order(order_type, symbol, modified_amount, modified_leverage) #, fee)
                    new_info = dict(
                        order_id=order.id, order_type=order_type,
                        amount=order.amount, volume= order.volume, leverage= order.leverage,
                    )
                except ValueError as e:
                    new_info = dict(error=str(e))

                orders_info[symbol].update(new_info)
                

        return orders_info, closed_orders_info


    def _get_prices(self, keys: List[str]=['close', 'open']) -> Dict[str, np.ndarray]:
        prices = {}

        for symbol in self.trading_symbols:
            get_price_at = lambda time: \
                self.original_simulator.price_at(symbol, time)[keys]

            if self.multiprocessing_pool is None:
                p = list(map(get_price_at, self.time_points))
            else:
                p = self.multiprocessing_pool.map(get_price_at, self.time_points)
            
            # for item in p:
            #     if item.shape != (2,):
            #         print ('shape= ', item.shape)
            #         print (item)
            prices[symbol] = np.array(p)

        return prices


    def _process_obs(self) -> np.ndarray:
        
        signal_features = pd.DataFrame(index=self.original_simulator.symbols_data[self.trading_symbols[0]].index)
        np.seterr(divide='ignore', invalid='ignore')

        for symbol in self.trading_symbols:
           df = self.original_simulator.symbols_data[symbol].copy()
           df['days']= df.index.day
           df['hours'] = df.index.hour
           df['returns']= np.log(df['close'].div(df['close'].shift(1)))
           df['Cdirection']=np.where(df["returns"] > 0, 1, 0)
           #df.dropna(inplace=True)
           df = df.add_prefix(symbol + ':')
           df = add_all_ta_features(df, open=symbol +':open', high=symbol+":high", low=symbol+":low", close=symbol+":close",\
                volume=symbol+":volume",fillna=True, colprefix=symbol + ':')
           signal_features = pd.concat([df,signal_features],join='inner',axis=1)

        #Deal with NaN values as a result of applying TA
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit_transform(signal_features)
        #df = pd.DataFrame(imputer, columns=signal_features.columns.values, index=signal_features.index)   
        return imputer #signal_features.values

        

    def _get_observation(self) -> Dict[str, np.ndarray]:
        features = self.signal_features[(self._current_tick - self.window_size+1):(self._current_tick+1)]

        orders = np.zeros(self.observation_space['orders'].shape)
        #print (orders.shape)
        for i, symbol in enumerate(self.trading_symbols):
            symbol_orders = self.simulator.symbol_orders(symbol)
            for j, order in enumerate(symbol_orders):
                orders[i, j] = [order.entry_price, order.amount, order.profit]

        observation = {
            'balance': np.array([self.simulator.balance]),
            'PnL': np.array([self.simulator.PnL]),
            'net_worth': np.array([self.simulator.net_worth]),
            'features': features,
            'orders': orders,
        }
        return observation


    def _calculate_reward(self) -> float:
        prev_nt = self.history[-1]['net_worth']
        current_nt = self.simulator.net_worth
        step_reward = current_nt - prev_nt
        return round (step_reward, 6)


    def _create_info(self, **kwargs: Any) -> Dict[str, Any]:
        info = {k: v for k, v in kwargs.items()}
        info['balance'] = self.simulator.balance
        info['PnL'] = self.simulator.PnL
        info['net_worth'] = self.simulator.net_worth
        #info['free_margin'] = self.simulator.free_margin
        #info['margin_level'] = self.simulator.margin_level
        return info


    def _get_modified_amount(self, symbol: str, amount: float) -> float:
        #si = self.simulator.symbols_info[symbol]
        amount = abs(amount)
        #amount = np.clip(amount, 0.01, 500)
        amount = round(amount , 2)
        #v = round(v / si.volume_step) * si.volume_step
        return np.exp(amount) * 10

    def _get_modified_leverage(self, symbol: str, leverage: float) -> int:
        symbol_info = self.simulator.symbols_info[symbol]
        leverage = expit(leverage) *100
        leverage = leverage * (symbol_info.max_leverage / 100)
        leverage = round(leverage)
        leverage = np.clip(leverage, symbol_info.min_leverage, symbol_info.max_leverage)
        return leverage


    def render(self, mode: str='human', **kwargs: Any) -> Any:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        if mode == 'advanced_figure':
            return self._render_advanced_figure(**kwargs)
        return self.simulator.get_state(**kwargs)


    def _render_simple_figure(
        self, figsize: Tuple[float, float]=(14, 6), return_figure: bool=False
    ) -> Any:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            ax.plot(self.time_points, close_price, c=symbol_color, marker='.', label=symbol)

            buy_ticks = []
            buy_error_ticks = []
            sell_ticks = []
            sell_error_ticks = []
            close_ticks = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol, {})
                if order and not order['hold']:
                    if order['order_type'] == OrderType.Buy:
                        if order['error']:
                            buy_error_ticks.append(tick)
                        else:
                            buy_ticks.append(tick)
                    else:
                        if order['error']:
                            sell_error_ticks.append(tick)
                        else:
                            sell_ticks.append(tick)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    close_ticks.append(tick)

            tp = np.array(self.time_points)
            ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            ax.yaxis.tick_left()
            if j < len(self.trading_symbols) - 1:
                ax = ax.twinx()

        fig.suptitle(
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"PnL: {self.simulator.PnL:.6f} ~ "
            f"Net Worth: {self.simulator.net_worth:.6f} ~ "
            )
        fig.legend(loc='right')

        if return_figure:
            return fig

        plt.show()


    def _render_advanced_figure(
            self, figsize: Tuple[float, float]=(1400, 600), time_format: str="%Y-%m-%d %H:%M:%S",
            return_figure: bool=False
        ) -> Any:

        fig = go.Figure()

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))
        get_color_string = lambda color: "rgba(%s, %s, %s, %s)" % tuple(color)

        extra_info = [
            f"balance: {h['balance']:.2f} {self.simulator.unit}<br>"
            f"PnL: {h['PnL']:.4f} {self.simulator.unit}<br>"
            f"net worth: {h['net_worth']:.4f} {self.simulator.unit}<br>"
            f"step reward: {h['step_reward'] if 'step_reward' in h else ''} {self.simulator.unit}<br>"
            # f"margin: {h['margin']:.6f}<br>"
            # f"free margin: {h['free_margin']:.6f}<br>"
            # f"margin level: {h['margin_level']:.6f}"
            for h in self.history
        ]
        extra_info = [extra_info[0]] * (self.window_size - 1) + extra_info

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            fig.add_trace(
                go.Scatter(
                    x=self.time_points,
                    y=close_price,
                    mode='lines+markers',
                    line_color=get_color_string(symbol_color),
                    opacity=1.0,
                    hovertext=extra_info,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                    # position=0.035*j
                ),
            })

            trade_ticks = []
            trade_markers = []
            trade_colors = []
            trade_sizes = []
            trade_extra_info = []
            trade_max_amount = max([
                h.get('orders', {}).get(symbol, {}).get('amount') or 0
                for h in self.history
            ])
            close_ticks = []
            close_extra_info = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol)
                if order and not order['hold']:
                    marker = None
                    color = None
                    size = 8 + 22 * (order['amount'] / trade_max_amount)
                    info = (
                        f"symbol: {order['symbol']}<br>"
                        f"order id: {order['order_id'] or ''}<br>"
                        f"hold probability: {order['hold_probability']:.4f}<br>"
                        f"hold: {order['hold']}<br>"
                        f"amount: {order['amount']:.2f}<br>"
                        f"Leverage: {order['leverage']}<br>"
                        f"volume: {order['volume']:.6f}<br>"
                        f"error: {order['error']}"
                    )

                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    info = []
                    for order in closed_orders:
                        info_i = (
                            f"order id: {order['order_id']}<br>"
                            f"order type: {order['order_type'].name}<br>"
                            f"close probability: {order['close_probability']:.4f}<br>"
                            f"order amount: {order['amount']:.2f}<br>"
                            f"leverage: {order['leverage']}<br>"
                            f"volume: {order['volume']:.6f}<br>"
                            f"profit: {order['profit']:.6f}<br>"
                            f"fee: {order['fee']:.2f}"
                        )
                        info.append(info_i)
                    info = '<br>---------------------------------<br>'.join(info)

                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[trade_ticks],
                    y=close_price[trade_ticks],
                    mode='markers',
                    hovertext=trade_extra_info,
                    marker_symbol=trade_markers,
                    marker_color=trade_colors,
                    marker_size=trade_sizes,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[close_ticks],
                    y=close_price[close_ticks],
                    mode='markers',
                    hovertext=close_extra_info,
                    marker_symbol='line-ns',
                    marker_color='black',
                    marker_size=8,
                    marker_line_width=2.5,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

        title = (
            f"Balance: {self.simulator.balance:.2f} {self.simulator.unit} ~ "
            f"PnL: {self.simulator.PnL:.4f} ~ "
            f"Net Worth: {self.simulator.net_worth:.4f} ~ "
            f"Closed Orders: {len(self.simulator.closed_orders)} ~ "
            f"Open Orders: {len(self.simulator.orders)}"
            # f"Margin: {self.simulator.margin:.6f} ~ "
            # f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            # f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.update_layout(
            title=title,
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        if return_figure:
            return fig

        fig.show()


    def close(self) -> None:
        plt.close()
