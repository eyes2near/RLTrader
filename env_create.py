from tensortrade.feed.core import DataFeed, Stream
from tensortrade.feed.core.base import IterableStream
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import dataset as datas
from ttextend import TensorTradeExtend
from tensortrade.env.default import stoppers,informers,renderers,rewards,actions
from tensortrade.env.generic import TradingEnv
from tensortrade.env.generic.components.renderer import AggregateRenderer
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

import random
import env_create

env_config = {
        "window_size":144, 
        'max_allowed_loss':0.2, 
        'reward_window_size':60, 
        'exchange':'binance', 
        'since':'2018-01', 
        'until':'2023-01',
        'timeframe':'1m',
        'total_partitions':11,
        'env_max_data_len':10000,
        'partition_overlap':288,
        'partition_id':0,
        'eval_len':10000,
        }

def train_env(config=env_config):
    total_partitions = config['total_partitions']
    partition_overlap = config['partition_overlap']
    partition_id = config['partition_id']
    eval_len = config['eval_len']
    if not hasattr(env_create,'train_dataframe'):
        print(config)
        full_dataframe = datas.load(config['exchange'], 
                                        timeframes=[config['timeframe']], 
                                        since=config['since'], 
                                        until=config['until'])[config['timeframe']]
        env_create.train_dataframe = full_dataframe[:-eval_len]
        env_create.eval_dataframe = full_dataframe[-eval_len:]
        total_data_len = len(env_create.train_dataframe)
        partition_len = (total_data_len - ((total_partitions-1)*partition_overlap))//total_partitions
        if partition_len >= total_partitions*partition_overlap:
            env_create.train_dataframe = env_create.train_dataframe[partition_id*partition_len:(partition_id+1)*partition_len+partition_overlap]
        print('Collector dataset between: ',
                env_create.train_dataframe['OpenTime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                env_create.train_dataframe['OpenTime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
              )
    total_data_len = len(env_create.train_dataframe)
    env_max_data_len = config['env_max_data_len']
    #环境数据从env_full_dataframe随机取出,最长env_max_data_len个数据，尽量长于total_partitions*partition_overlap且尽量更长
    if total_data_len <= total_partitions*partition_overlap:
        dataset = env_create.train_dataframe
    if total_data_len>env_max_data_len:
        begin_idx = random.randint(0,total_data_len-env_max_data_len)
        dataset = env_create.train_dataframe[begin_idx:begin_idx+env_max_data_len]
    print('Current env samples data between: ', dataset['OpenTime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'), dataset['OpenTime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'))
    commission = 0.002  # TODO: adjust according to your commission percentage, if present
    price = TensorTradeExtend.random_price_stream(dataset['Low'], dataset['High'], 'float', USD.precision).rename("USD-BTC")
    options = ExchangeOptions(commission=commission)
    price_close = IterableStream(dataset['Close'],dtype='float').rename("USD-BTC_close")
    exchange = Exchange("Binance", service=execute_order, options=options)(price,price_close)

    # Instruments, Wallets and Portfolio
    cash = Wallet(exchange, 100000 * USD)  # This is the starting cash we are going to use
    asset = Wallet(exchange, 5 * BTC)  # And we will start owning 5 stocks of BTC
    portfolio = Portfolio(USD, [cash, asset])

    # Renderer feed
    renderer_feed = DataFeed([
        Stream.source(list(dataset["OpenTime"])).rename("date"),
        Stream.source(list(dataset["Open"]), dtype="float").rename("open"),
        Stream.source(list(dataset["High"]), dtype="float").rename("high"),
        Stream.source(list(dataset["Low"]), dtype="float").rename("low"),
        Stream.source(list(dataset["Close"]), dtype="float").rename("close"),
        Stream.source(list(dataset["Volume"]), dtype="float").rename("volume")
    ])

    #dataset = addIndicators(dataset)
    features = []
    for c in dataset.columns[1:]:
        s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
        features += [s]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = actions.BSH(cash=cash, asset=asset)
    action_scheme.portfolio = portfolio

    min_periods = config["window_size"]-1

    observer = TensorTradeExtend.random_price_attached_observer(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=config["window_size"],
        min_periods=min_periods
    )

    stopper = stoppers.MaxLossStopper(
        max_allowed_loss=config["max_allowed_loss"]
    )
    renderer_list = config["renderer"] if "renderer" in config else renderers.EmptyRenderer()

    if isinstance(renderer_list, list):
        for i, r in enumerate(renderer_list):
            if isinstance(r, str):
                renderer_list[i] = renderers.get(r)
        renderer = AggregateRenderer(renderer_list)
    else:
        if isinstance(renderer_list, str):
            renderer = renderers.get(renderer_list)
        else:
            renderer = renderer_list

    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=stopper,
        informer=informers.TensorTradeInformer(),
        renderer=renderer,
        min_periods=min_periods,
        random_start_pct=0.8,
    )
    env.price = price
    ret = suite_gym.wrap_env(env)
    ret.tf = toTF(ret)
    return ret

def eval_env(config=env_config):
    eval_len = config['eval_len']
    if not hasattr(env_create,'eval_dataframe'):
        print(config)
        full_dataframe = datas.load(config['exchange'], 
                                        timeframes=[config['timeframe']], 
                                        since=config['since'], 
                                        until=config['until'])[config['timeframe']]
        env_create.eval_dataframe = full_dataframe[-eval_len:]
    dataset = env_create.eval_dataframe
    commission = 0.002  # TODO: adjust according to your commission percentage, if present
    price = TensorTradeExtend.random_price_stream(dataset['Low'], dataset['High'], 'float', USD.precision).rename("USD-BTC")
    price_close = IterableStream(dataset['Close'],dtype='float').rename("USD-BTC_close")
    options = ExchangeOptions(commission=commission)
    exchange = Exchange("Binance", service=execute_order, options=options)(price,price_close)

    # Instruments, Wallets and Portfolio
    cash = Wallet(exchange, 100000 * USD)  # This is the starting cash we are going to use
    asset = Wallet(exchange, 5 * BTC)  # And we will start owning 5 stocks of BTC
    portfolio = Portfolio(USD, [cash, asset])

    # Renderer feed
    renderer_feed = DataFeed([
        Stream.source(list(dataset["OpenTime"])).rename("date"),
        Stream.source(list(dataset["Open"]), dtype="float").rename("open"),
        Stream.source(list(dataset["High"]), dtype="float").rename("high"),
        Stream.source(list(dataset["Low"]), dtype="float").rename("low"),
        Stream.source(list(dataset["Close"]), dtype="float").rename("close"),
        Stream.source(list(dataset["Volume"]), dtype="float").rename("volume")
    ])

    #dataset = addIndicators(dataset)
    
    features = []
    for c in dataset.columns[1:]:
        s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
        features += [s]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = actions.BSH(cash=cash, asset=asset)
    action_scheme.portfolio = portfolio

    min_periods = config["window_size"]-1

    observer = TensorTradeExtend.random_price_attached_observer(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=config["window_size"],
        min_periods=min_periods
    )

    stopper = stoppers.MaxLossStopper(
        max_allowed_loss=config["max_allowed_loss"]
    )
    renderer_list = config["renderer"] if "renderer" in config else renderers.EmptyRenderer()

    if isinstance(renderer_list, list):
        for i, r in enumerate(renderer_list):
            if isinstance(r, str):
                renderer_list[i] = renderers.get(r)
        renderer = AggregateRenderer(renderer_list)
    else:
        if isinstance(renderer_list, str):
            renderer = renderers.get(renderer_list)
        else:
            renderer = renderer_list

    env = TradingEnv(
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        observer=observer,
        stopper=stopper,
        informer=informers.TensorTradeInformer(),
        renderer=renderer,
        min_periods=min_periods,
        random_start_pct=0.8,
    )
    env.price = price
    ret = suite_gym.wrap_env(env)
    ret.tf = toTF(ret)
    return ret

def toTF(env):
    return TFPyEnvironment(env)