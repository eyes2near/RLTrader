# 這個是給trainer提供的facade service，所有對後端collector服務的調用全部由DriverCollector來進行。
# 使用一組後端collector地址進行初始化，傳入observers作爲回調，observe(episode_id, traj)
# collector_batch 指明了在每一個後端collector服務上批量進行的用於收集的env的數量。

import pickle
import time
import grpc
import collector_pb2
import collector_pb2_grpc
from tf_agents.trajectories import trajectory
import tensorflow as tf
import numpy as np
import redis

redis_db = 0             # 数据库编号
redis_password = None    # Redis密码（如果有的话）

class DriverCollector:

    def __init__(self, collector_addrs, observers, collector_batch = 5) -> None:
        self.collector_addrs = collector_addrs if isinstance(collector_addrs,list) else [collector_addrs]
        self.observers = observers if isinstance(observers,list) else [observers]
        self.collector_batch = collector_batch
        self.num_collectors=len(self.collector_addrs)
        self.channels = []
        self.stubs = []
        for addr in self.collector_addrs:
            max_message_length = 300 * 1024 * 1024  # Set the max message length to 300 MB
            options = [('grpc.max_send_message_length', max_message_length),
                ('grpc.max_receive_message_length', max_message_length)]
            channel = grpc.insecure_channel(target = addr, options=options)
            stub = collector_pb2_grpc.CollectServiceStub(channel)
            self.channels.append(channel)
            self.stubs.append(stub)
        resp = stub.envspecs(collector_pb2.EnvSpecReq())
        self.action_spec=pickle.loads(resp.action_spec)
        self.time_step_spec=pickle.loads(resp.time_step_spec)
        self.observation_spec=pickle.loads(resp.observation_spec)

    def update_collect_policy(self, counter, policy, redis_addr = "127.0.0.1:6379"):
        parts = redis_addr.split(':')
        host = parts[0]  # 第一部分是 host
        port = int(parts[1])  # 第二部分是 port，将其转换为整数
        vars = policy.variables()
        #set to redis
        redis_conn = redis.StrictRedis(host=host, port=port, db=redis_db, password=redis_password)
        cbytes = pickle.dumps(counter)
        vbytes = pickle.dumps(vars)
        print("var size->", len(vbytes)/1024/1024, "MB")
        redis_conn.set("counter", cbytes)
        redis_conn.set("variables", vbytes)
        redis_conn.close()
        #sync variables to collectors
        futures = []
        for stub in self.stubs:
            futures.append(stub.notify_policy_updated.future(
                collector_pb2.NotifyPolicyUpdatedReq(
                    redis_addr=redis_addr
                )
            ))
        for f in futures:
            f.result()

    def collect(self, num_episodes):
        #更新policy完毕之后，开始收集
        num_to_collect = num_episodes
        workers = []
        nums_ep_per_worker = []
        if num_to_collect < self.num_collectors:
            workers = self.stubs[0:num_to_collect]
            nums_ep_per_worker = [1]*num_to_collect
        elif num_to_collect < self.num_collectors*self.collector_batch:
            workers = self.stubs
            nums_ep_per_worker = [num_to_collect//self.num_collectors]*self.num_collectors
            for i in range(num_to_collect%self.num_collectors):
                nums_ep_per_worker[i]+=1
        else:
            workers = self.stubs
            nums_ep_per_worker=[self.collector_batch]*self.num_collectors
        ep_id_cursor = 0
        while num_to_collect > 0:
            collector_resps = []
            for id,stub in enumerate(workers):
                nums= nums_ep_per_worker[id]
                collector_resps.append(stub.collect(collector_pb2.CollectReq(num_episodes=nums)))
                num_to_collect-=nums
                if num_to_collect <= 0:
                    break
            if num_to_collect >= self.num_collectors and num_to_collect < self.collector_batch*self.num_collectors:
                workers = self.stubs
                nums_ep_per_worker = [num_to_collect//self.num_collectors]*self.num_collectors
                for i in range(num_to_collect%self.num_collectors):
                    nums_ep_per_worker[i]+=1
            elif num_to_collect > 0  and num_to_collect < self.num_collectors:
                workers = self.stubs[0:num_to_collect]
                nums_ep_per_worker = [1]*num_to_collect
            batch_done = False
            while not batch_done:
                batch_done = True
                remaining_resps = []
                for resps in collector_resps:
                    try:
                        resp = next(resps)
                    except StopIteration:
                        continue
                    remaining_resps.append(resps)
                    batch_done = False
                    eid = ep_id_cursor
                    ep_id_cursor+=1
                    traj_obs = {
                        "market": tf.reshape(resp.obs_market, [1,-1,]+ self.observation_spec["market"].shape.as_list()),
                        "stateful": tf.reshape(resp.obs_stateful, [1,-1,]+ self.observation_spec["stateful"].shape.as_list())
                    }
                    traj = trajectory.Trajectory(
                        tf.constant([resp.types],dtype=tf.int32),
                        traj_obs,
                        tf.constant([resp.actions],dtype=tf.int64),
                        pickle.loads(resp.ps_infos),
                        tf.constant([resp.next_types],dtype=tf.int32),
                        tf.constant([resp.rewards],dtype=tf.float32),
                        tf.constant([resp.discounts],dtype=tf.float32),
                    )
                    for observer in self.observers:
                        observer(eid, traj)
                collector_resps = remaining_resps