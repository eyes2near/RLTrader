from typing import Iterable
from tensortrade.feed.core.base import Stream,IterableStream
from tensortrade.env.generic import TradingEnv
from tensortrade.env.default import observers
import random
from gym.spaces import Box,Dict
import numpy as np

class TensorTradeExtend:
    
    @staticmethod
    def random_price_stream(lower: "Iterable", upper: "Iterable", dtype: str = None, precision: int = -1, length=1000):
        s = IterableStream(list(zip(lower,upper)), dtype)
        lower,upper = s.current
        if precision >= 0:
            s.current = round(random.uniform(lower, upper), precision)
        else:
            s.current = random.uniform(lower, upper)
        og = s.generator
        #预先生成length个ratio,用以生成price stream, price = low + ratio * (high - low) 
        s.price_ratio_it = iter([random.uniform(0.0,1.0) for _ in range(length)])
        def newg(g):
            try:
                while True:
                    l,u = next(g)
                    r = next(s.price_ratio_it)
                    if precision >= 0:
                        v = round(l + r*(u-l), precision)
                    else:
                        v = l + r*(u-l)
                    #print('luv = ',l,u,v)
                    yield v
            except StopIteration:
                return
        s.generator = newg(og)
        
        def new_reset(random_start=0):
            #check context reset flags to avoid duplicate resets
            if hasattr(Stream.stream_thread_local, "reset_member"):
                sid = id(s)
                if sid in Stream.stream_thread_local.reset_member:
                    return
                else:
                    Stream.stream_thread_local.reset_member.add(sid)
            #print('reset at index->', random_start)
            s.price_ratio_it = iter([random.uniform(0.0,1.0) for _ in range(length)])
            if random_start != 0:
                s._random_start = random_start
            if s.is_gen:
                s.generator = newg(s.gen_fn())
            else:
                s.generator = newg(iter(s.iterable[s._random_start:]))
            s.stop = False
            try:
                s.current = next(s.generator)
            except StopIteration:
                s.stop = True
            s.forward()

        sforward = s.forward
        def new_forward():
            s.previous = sforward()
            return s.previous
        s.forward = new_forward
        s.reset = new_reset
        if dtype:
            s = s.astype(dtype)
        s.forward()
        return s

    @staticmethod
    def random_price_attached_observer(portfolio, feed, renderer_feed, window_size, min_periods):
        observer = observers.TensorTradeObserver(
            portfolio=portfolio,
            feed=feed,
            renderer_feed=renderer_feed,
            window_size=window_size,
            min_periods=min_periods
        )
        #觀察空間包括{"market": olhcv_data, "stateful": stateful_data, "news": news_data}
        #其中stateful_data包括當前目標貨幣的價格，usdt賬戶，目標貨幣的市值(以usdt計價)
        #news_data是抓取的相關新聞，經過chatgpt對一系列問題的打分。
        market_space = observer._observation_space
        stateful_data = Box(low=0.0, high=np.inf, shape=(3,), dtype=np.float32)
        #TODO:news 以後再實現
        #news_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        observer._observation_space = Dict({
            "market": market_space,
            "stateful": stateful_data,
            #"news": news_space
        })

        current_observe = observer.observe
        def new_observe(env: 'TradingEnv') -> np.array:
            obs = current_observe(env)
            balances = portfolio.total_balances
            usdt_balance = balances[0].as_float()
            coin_balance = balances[1].as_float()
            # print("balance->",balances[0], balances[1])
            #add price to observation
            price = env.price.previous
            stateful_obs = np.array([price, usdt_balance, coin_balance * price]).astype(observer._observation_dtype)
            return {"market": obs, "stateful": stateful_obs}
        
        observer.observe = new_observe
        return observer
    
