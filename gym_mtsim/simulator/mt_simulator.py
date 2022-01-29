from typing import List, Tuple, Dict, Any, Optional

import os
import pickle
from datetime import datetime, timedelta
import ssl, json, time
import numpy as np
import pandas as pd
import urllib.request

from gym_mtsim.data.cdd import CryptoDataDownload
from ..metatrader import SymbolInfo
#from ..envs import SymbolInfo this was my mistake, you should not import env because env is also importing simulator
from .order import OrderType, Order
from .exceptions import SymbolNotFound, OrderNotFound


class MtSimulator:

    def __init__(
            self, symbols_filename :str, unit: str='USD', balance: float=10000.,
            stop_out_level: float=0.2, hedge: bool=True, 
            dataset: str= 'train'
        ) -> None:

        self.unit = unit
        self.balance = balance
        self.original_balance = balance
        self.PnL = 0.
        self.net_worth = balance
        self.stop_out_level = stop_out_level
        self.hedge = hedge
        self.dataset = dataset
        self.symbols_info: Dict[str, SymbolInfo] = {}
        self.symbols_data: Dict[str, pd.DataFrame] = {}
        self.orders: List[Order] = []
        self.closed_orders: List[Order] = []
        self.current_time: datetime = NotImplemented

        if symbols_filename:
            if not self.load_symbols(symbols_filename, dataset):
                raise FileNotFoundError(f"file '{symbols_filename}' not found")


    # @property
    # def free_margin(self) -> float:
    #     return self.equity - self.margin


    # @property
    # def margin_level(self) -> float:
    #     margin = round(self.margin, 6)
    #     if margin == 0.:
    #         return np.inf
    #     return self.equity / margin
    @property
    def balance_level(self) -> float:
        return self.balance / self.original_balance

    def download_data(self, symbols: List[str] ) -> None:
        cdd= CryptoDataDownload()
       
        #Loop to remove duplicates and store data       
        for symbol in symbols:
            df = cdd.fetch('Bitfinex', 'USD', symbol.split('/')[0], '1h')
            df = df.drop(df[df.index.duplicated()].index)
            df = df.drop(df[df.duplicated()].index)
            self.symbols_data[symbol] = df
            self.symbols_info[symbol] = SymbolInfo(symbol,'Bitfinex', 5, 300)
        
        #time to find intersection between ALL symbols, store the first frame
        first_df = self.symbols_data[symbols[0]]
        
        for symbol in symbols[1:]:
            intersection = first_df.index.intersection(self.symbols_data[symbol].index)
            first_df = first_df.loc[intersection]

        #Apply the filter to all store symbols   
        for symbol in symbols:
            self.symbols_data[symbol] = self.symbols_data[symbol].loc[first_df.index]
       
        print ('Download completed')

        # #download data from LunarCrush
        # ssl._create_default_https_context = ssl._create_unverified_context
        # api_key = "qgua98oy6jisudo4o0u3s"
        
        # for symbol in symbols:
        #     url = "https://api.lunarcrush.com/v2?data=assets&key=" + api_key + "&symbol="+ symbol.split('/')[0] + '&data_points=720'
        #     assets = json.loads(urllib.request.urlopen(url).read())
        #     df = pd.json_normalize(assets, record_path=['data','timeSeries'])
        #     df.index = pd.to_datetime(df['time'], unit='s')
        #     time.sleep(5)

        #     # for _ in range (20):
        #     #     url = "https://api.lunarcrush.com/v2?data=assets&key=" + api_key + "&symbol="+ symbol.split('/')[0] + '&data_points=720' + '&end=' + str(int(pd.Timestamp(df.index[0]).timestamp()))
        #     #     assets = json.loads(urllib.request.urlopen(url).read())
        #     #     temp_df = pd.json_normalize(assets, record_path=['data','timeSeries'])
        #     #     temp_df.index = pd.to_datetime(temp_df['time'], unit='s')
        #     #     df = df.append(temp_df)
        #     #     df = df.sort_index()
        #     #     time.sleep(5)
            
        #     df['days']= df.index.day
        #     df['hours'] = df.index.hour
        #     df['returns']= np.log(df['close'].div(df['close'].shift(1)))
        #     df['Cdirection']=np.where(df["returns"] > 0, 1, 0)
        #     df = df.drop(['asset_id','search_average','time'], axis=1)

        #     self.symbols_data[symbol] = df
        #     self.symbols_info[symbol] = SymbolInfo(symbol,'Bitfinex', 5, 300, 0.01)



    def save_symbols(self, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump((self.symbols_info, self.symbols_data), file)
            print ('Data saved.')


    def load_symbols(self, filename: str, dataset: str= 'train') -> bool:
        
        if not os.path.exists(filename):
            return False
        with open(filename, 'rb') as file:
            self.symbols_info, self.symbols_data = pickle.load(file)
            
            for symbol in self.symbols_data:
                data_split = int(len(self.symbols_data[symbol]) * 0.7)
                if dataset == 'train':
                    self.symbols_data[symbol] = self.symbols_data[symbol] [:data_split]
                else:
                    self.symbols_data[symbol] = self.symbols_data[symbol] [data_split:]
        print (f'Dataset: {dataset} Loaded')
        return True


    def tick(self, delta_time: timedelta=timedelta()) -> None:
        self._check_current_time()

        self.current_time += delta_time
        self.net_worth = self.balance
        self.PnL = 0.
        #closed_orders_info = {symbol: [] for symbol in self.orders}

        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)['close']
            self._update_order_profit(order)
            self.net_worth += order.amount + order.profit
            self.PnL += order.profit
            #check if order profit is <= the original order amount
            if order.amount + order.profit <= 0:
                self.close_order(order)
                self.PnL -= order.profit
            
                
        while self.balance_level < self.stop_out_level and len(self.orders) > 0: #self.margin_level < self.stop_out_level and 
            most_unprofitable_order = min(self.orders, key=lambda order: order.profit)
            self.close_order(most_unprofitable_order)

        if self.balance < 0.:
            self.balance = 0.
            #self.equity = self.balance


    def nearest_time(self, symbol: str, time: datetime) -> datetime:
        df = self.symbols_data[symbol]
        if time in df.index:
            return time
        try:
            i = df.index.get_loc(time, method='ffill')
        except KeyError:
            i = df.index.get_loc(time, method='bfill')
        return df.index[i]


    def price_at(self, symbol: str, time: datetime) -> pd.Series:
        df = self.symbols_data[symbol]
        time = self.nearest_time(symbol, time)
        return df.loc[time]


    def symbol_orders(self, symbol: str) -> List[Order]:
        symbol_orders = list(filter(
            lambda order: order.symbol == symbol, self.orders
        ))
        return symbol_orders


    def create_order(self, order_type: OrderType, symbol: str, amount: float, leverage :int) -> Order:
        self._check_current_time()

        order_id = len(self.closed_orders) + len(self.orders) + 1
        entry_time = self.current_time
        entry_price = self.price_at(symbol, entry_time)['close']
        exit_time = entry_time
        exit_price = entry_price

        order = Order(
            order_id, order_type, symbol, amount, leverage,
            entry_time, entry_price, exit_time, exit_price
        )
        
        #removed because profit = 0 and fee = 0 at this time and there are no fees to deduct
        #self._update_order_profit(order)
        
        if order.amount > self.balance + order.profit:
            raise ValueError(
                f"Low free balance (order amount={order.amount},"
                f"Balance={self.balance})"
            )

        #self.PnL += order.profit do NOT do this because profit is always =0 when u create a new order exit_time = entry_time
        self.balance -= order.amount
        self.orders.append(order)
        return order


    def close_order(self, order: Order) -> float:
        self._check_current_time()
        if order not in self.orders:
            raise OrderNotFound("order not found in the order list")

        order.exit_time = self.current_time
        order.exit_price = self.price_at(order.symbol, order.exit_time)['close']
        self._update_order_profit(order)

        self.balance += order.amount + order.profit
        #self.PnL -= order.profit
        # print ("before:", self.PnL)
        # self.PnL = self.PnL - abs(order.profit)
        # print ('after:', self.PnL)

        order.closed = True
        self.orders.remove(order)
        self.closed_orders.append(order)
        return order.profit


    def get_state(self) -> Dict[str, Any]:
        orders = []
        for order in reversed(self.closed_orders + self.orders):
            orders.append({
                'Id': order.id,
                'Symbol': order.symbol,
                'Type': order.type.name,
                'Amount': order.amount,
                'Leverage': order.leverage,
                'Entry Time': order.entry_time,
                'Entry Price': order.entry_price,
                'Exit Time': order.exit_time,
                'Exit Price': order.exit_price,
                'Profit': order.profit,
                'Fee': order.fee,
                'Closed': order.closed,
            })
        orders_df = pd.DataFrame(orders)

        return {
            'current_time': self.current_time,
            'balance': self.balance,
            'PnL': self.PnL,
            'Net Worth': self.net_worth,
            # 'margin': self.margin,
            # 'free_margin': self.free_margin,
            # 'margin_level': self.margin_level,
            'orders': orders_df,
        }


    def _update_order_profit(self, order: Order) -> None:
        diff = order.exit_price - order.entry_price
        order.volume = (order.amount * order.leverage) / order.entry_price
        local_profit = order.volume * (order.type.sign * diff)
        if local_profit > 0:
            order.profit = round (local_profit * 0.9, 4) #take 10% revenue share as comission
            order.fee = round (local_profit * 0.1, 4)
        else:
            order.profit = round (local_profit , 4)
            order.fee = 0 


    def _get_unit_ratio(self, symbol: str, time: datetime) -> float:
        symbol_info = self.symbols_info[symbol]
        if self.unit == symbol_info.currency_profit:
            return 1.

        if self.unit == symbol_info.currency_margin:
            return 1 / self.price_at(symbol, time)['close']

        currency = symbol_info.currency_profit
        unit_symbol_info = self._get_unit_symbol_info(currency)
        if unit_symbol_info is None:
            raise SymbolNotFound(f"unit symbol for '{currency}' not found")

        unit_price = self.price_at(unit_symbol_info.name, time)['Close']
        if unit_symbol_info.currency_margin == self.unit:
            unit_price = 1. / unit_price

        return unit_price


    def _get_unit_symbol_info(self, currency: str) -> Optional[SymbolInfo]:  # Unit/Currency or Currency/Unit
        for info in self.symbols_info.values():
            if currency in info.currencies and self.unit in info.currencies:
                return info
        return None


    def _check_current_time(self) -> None:
        if self.current_time is NotImplemented:
            raise ValueError("'current_time' must have a value")


