import numpy as np

def get_initial_mid_spread(data):
    initial_bid = data.iloc[0]["bid_price_1"]
    initial_ask = data.iloc[0]["ask_price_1"]
    return (initial_bid + initial_ask) / 2
