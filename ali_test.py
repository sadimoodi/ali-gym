import pytz
from datetime import datetime, timedelta
import numpy as np
from gym_mtsim import MtEnv, MtSimulator, FOREX_DATA_PATH, OrderType, ALI_DATA_PATH


sim = MtSimulator(
    unit='USD',
    balance=10000.,
    leverage=100.,
    stop_out_level=0.2,
    hedge= True,
    symbols_filename=ALI_DATA_PATH
)

env = MtEnv(
    original_simulator=sim,
    trading_symbols=['BTC/USD', 'ETH/USD', 'LTC/USD'],
    window_size=10,
    # time_points=[desired time points ...],
    hold_threshold=0.5,
    close_threshold=0.5,
    fee=lambda symbol: {
        'BTC/USD': max(0., np.random.normal(0.0007, 0.00005)),
        'ETH/USD': max(0., np.random.normal(0.0002, 0.00003)),
        'LTC/USD': max(0., np.random.normal(0.02, 0.003)),
    }[symbol],
    symbol_max_orders=2,
    multiprocessing_processes=0
)

# print("env information:")

# for symbol in env.prices:
#     print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

# print("> signal_features.shape:", env.signal_features.shape)
# print("> features_shape:", env.features_shape)



observation = env.reset()

#action = env.action_space.sample()

while True:
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    
    if done:
        # print(info)
        print(
            f"balance: {info['balance']}, equity: {info['equity']},\n"
            f"step_reward: {info['step_reward']}"
        )
        break

state = env.render()

env.render('advanced_figure', time_format="%Y-%m-%d %H:%M:%S")
#print (state['orders'])