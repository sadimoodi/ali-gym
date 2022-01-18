from typing import Tuple

from .interface import MtSymbolInfo


class SymbolInfo:

    def __init__(self, symbol: str, market: str, min_leverage: int, max_leverage: int, amount_step: float) -> None:
        #print ('entered the constructor of SymbolInfo')
        self.name: str = symbol
        self.market: str = market

        self.currency_margin: str = symbol.split('/')[0]
        self.currency_profit: str = symbol.split('/')[1]
        self.currencies: Tuple[str, ...] = tuple(set([self.currency_margin, self.currency_profit]))

        #self.trade_contract_size: float = info.trade_contract_size
        #self.margin_rate: float = 1.0  # MetaTrader info does not contain this value!

        self.min_leverage: float = min_leverage
        self.max_leverage: float = max_leverage
        self.amount_step: float = amount_step


    def __str__(self) -> str:
        return f'{self.market}:{self.name}'


    # def _get_market(self, info: MtSymbolInfo) -> str:
    #     mapping = {
    #         'forex': 'Forex',
    #         'crypto': 'Crypto',
    #         'stock': 'Stock',
    #     }

    #     root = info.path.split('\\')[0]
    #     for k, v in mapping.items():
    #         if root.lower().startswith(k):
    #             return v

    #     return root
