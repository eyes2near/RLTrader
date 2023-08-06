import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
import tensorflow as tf
import grpc
import collector_pb2
import collector_pb2_grpc
from concurrent import futures
import argparse
from tf_agents.environments import suite_gym
import traceback
import pickle
import agent_create
from tf_agents.specs import tensor_spec
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import numpy as np
import time
import multiprocessing
from queue import Queue

import signal

parser = argparse.ArgumentParser()
parser.add_argument("--num_worker", type=int, help="number of workers to step env")
parser.add_argument("--num_env", type=int, help="number of envs to step in a worker")
parser.add_argument("--max_steps", type=int, help="max steps env can take")
parser.add_argument("--env_creator", type=str, help="env creator module and function")
parser.add_argument("--port", type=str, help="port to start service")
args = parser.parse_args()

def worker_initializer():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class CollectorServicer(collector_pb2_grpc.CollectServiceServicer):

    def worker(id, to_worker_queue, to_driver_queue, env_creator, env_config_template, num_env):
        envs = []
        partition_begin = num_env*id
        for env_id in range(num_env):
            env_cfg = env_config_template.copy()
            env_cfg['partition_id']=partition_begin+env_id
            env = env_creator(env_cfg)
            if not isinstance(env, TFPyEnvironment):
                env = TFPyEnvironment(env)
            envs.append(env)

        while True:
            try:
                msg_type, msg_data = to_worker_queue.get()
            except InterruptedError:
                break
            except KeyboardInterrupt:
                break
            #print('process recv -> ', msg_type, msg_data)
            #发送类型2，代表执行step
            if msg_type == 2:
                episode_id, env_id, action = msg_data
                print('collector process, step -> ', msg_data)
                if action == -860723:
                    ts = envs[env_id].reset()
                else:
                    ts = envs[env_id].step(action)
                to_driver_queue.put((episode_id, env_id, action, ts))
                continue
            #发送类型1，代表需要获得环境的spec
            if msg_type == 1:
                to_driver_queue.put((env.observation_spec(),env.action_spec(),env.time_step_spec()))
                continue
            #发送类型0，代表terminate
            if msg_type == 0:
                print('worker terminated.')
                to_driver_queue.put(())
                break
        

    def __init__(self, env_creator, num_worker, num_env, max_steps) -> None:
        #开启worker进程
        self.num_worker = num_worker
        self.num_env = num_env
        self.worker_queues = []
        self.worker_results = []
        self.processes = []
        for worker_id in range(num_worker):
            to_worker_queue = multiprocessing.Queue()
            to_driver_queue = multiprocessing.Queue()
            self.worker_queues.append((to_worker_queue,to_driver_queue))
            p = multiprocessing.Process(target=CollectorServicer.worker,args=(worker_id, to_worker_queue, to_driver_queue, env_creator,env_creator.config_template, num_env,))
            p.start()
            self.processes.append(p)
        self.max_steps = max_steps
        #获取环境spec
        to_worker_queue.put((1,()))
        self.obs_spec,self.act_spec,self.ts_spec = to_driver_queue.get()
        self.train_step_counter,agent = agent_create.ppo(self.obs_spec,self.act_spec,self.ts_spec)
        self.collect_policy = agent.collect_policy
        self.stateful = agent.collect_policy.get_initial_state(batch_size=1) != ()
        self.workload_p = 0.7
        import threading
        self.step_queue = Queue()
        def step_queue_processor(id, driver_queue):
            to_driver_queue = self.worker_queues[id][1]
            while True:
                got = to_driver_queue.get()
                print('queue thread ', id, ' got -> ', got)
                if got:
                    driver_queue.put((id, *got))
                else:
                    break

        for id in range(num_worker):
            threading.Thread(target=step_queue_processor, args=(id,self.step_queue)).start()
        def handler(signum, frame):
            if hasattr(handler,'_called'):
                return
            else:
                handler._called = 1
            if signum == signal.SIGINT:
                print('Terminating Collect Server...')
                for q,_ in self.worker_queues:
                    q.put((0,()))
                for p in self.processes:
                    p.join()
                os.kill(os.getpid(), signal.SIGTERM)

        # 注册信号处理器
        signal.signal(signal.SIGINT, handler)
        super().__init__()

    def update_policy(self, request, context):
        self.train_step_counter = pickle.loads(request.train_step_counter)
        variables = pickle.loads(request.variables)
        for variable, value in zip(
            tf.nest.flatten(self.collect_policy.variables()), tf.nest.flatten(variables)):
            variable.assign(value)
        return collector_pb2.UpdatePolicyResp()
    
    def collect(self, request, context):
        #排干step_queue
        while not self.step_queue.empty():
            self.step_queue.get()
        try:
            num_to_collect = request.num_episodes
            print('request to collect num_episodes -> ',num_to_collect)
            num_collected = 0
            available_worker_envs = set()
            #working_envs = set()
            last_timesteps = set()
            batched_tss = []
            batched_epids = []
            policy_step_states = []
            batched_worker_envs = []
            episode_id_cursor = 0
            #episode_worker_env_mappings = [[-1]*self.num_env] * self.num_worker #用来追踪当前(worker,env)对应的episode

            #当episode没有完结之前，存储已经产生的timestep和policy_step数据，位置索引就是episode_id
            #格式: [observations,discounts,rewards,pinfos,actions,types,next_types]
            episode_buffer = []
            for env_id in range(num_env):
                for worker_id in range(num_worker):
                    #等同于按照env_id排序，增加worker维度的并行
                    available_worker_envs.add((env_id,worker_id))
            while num_to_collect > 0:
                while len(available_worker_envs) > 0:
                    env_id,worker_id=available_worker_envs.pop()
                    #开始一个episode，发送action=-860723
                    to_worker_queue,_ = self.worker_queues[worker_id]
                    to_worker_queue.put((2, (episode_id_cursor, env_id, -860723)))
                    episode_id_cursor+=1
                    episode_buffer.append(
                        [[],#observations,0
                        tf.constant([],dtype=tf.float32),#discounts,1
                        tf.constant([],dtype=tf.float32),#rewards,2
                        (),#ps_infos,3
                        tf.constant([],dtype=tf.int64),#actions,4
                        tf.constant([],dtype=tf.int32),#step_types,5
                        tf.constant([],dtype=tf.int32),#next_step_types,6
                        self.collect_policy.get_initial_state(batch_size=1)#policy step states用来给collect_policy的action()使用,7
                        ])
                    num_to_collect-=1
                    if num_to_collect == 0:
                        break
            #收集返回的timesteps
            while num_collected < request.num_episodes:
                iteration_timesteps = []
                num_not_collected = request.num_episodes-num_collected
                act_batch_min_size = int(self.workload_p * min(self.num_worker * self.num_env,num_not_collected))
                act_batch_min_size = int(max(act_batch_min_size, 1))
                act_batch_min_size = num_not_collected if num_not_collected <= self.num_worker else act_batch_min_size
                print('take batch steps -> ', act_batch_min_size)
                while len(iteration_timesteps) < act_batch_min_size:
                    iteration_timesteps.append(self.step_queue.get())
                print('batch taken!')
                for item in iteration_timesteps:
                    worker_id,episode_id,env_id,action,ts = item
                    # 这个time_step可能是：
                    # (1) episode的起始帧，此时action是-860723并且episode_buffer中无数据
                    #     此时要：
                    #           1. 填充episode_buffer(observation,step_type=first)
                    #           2. 用collect_policy计算observation对应的policy_step,填充episode_buffer(action,ps_info,ps_state)
                    #           3. 把当前的(env_id,action)提交给worker，让worker去step生成下一个ts
                    #
                    # (2) episode的中间帧(注意：type=last并不代表不再step了，此时为了收集完整的经验路径，还要做一个额外的收尾的step)
                    #     因此，这个type=last的帧我们也当成中间帧
                    #     此时要：
                    #           1. 检查ts的type，有两种情况 type = mid 或者 last，不同情况下有不同的处理
                    #           2. 检查episode_buffer中此episode的数据长度，如果已经达到max_steps-1则将此ts标记为limit_as_last
                    #           3. 填充episode_buffer(
                    #                       observation,
                    #                       discount,
                    #                       reward,
                    #                       step_type=mid或者last(limit_as_last也按照last放入),
                    #                       next_step_type=step_type,
                    #                  )
                    #           4. 用collect_policy计算observation对应的policy_step,填充episode_buffer(action,ps_info)
                    #           5. 根据type=mid | last | limit_as_last决定要提交给worker的action
                    #              当 mid | limit_as_last 时: action = policy_step
                    #              当 last 时: action = -860723       
                    #              把当前的(env_id,action)提交给worker，让worker去step生成下一个ts
                    #
                    # (3) episode的结尾帧，此时上一个ts_type是last
                    #     此时要：
                    #           1. 填充episode_buffer(discounts,rewards,nex_step_type=first)
                    #           2. 利用episode_buffer中此episode的数据构建grpc调用结果yield发给调用端
                    #           3. 清空episode_buffer中此episode数据,节约内存
                    #           4. 查看num_to_collect是不是有新的episode需要收集，如果有，搬运118-132行同款代码

                    ebuf = episode_buffer[episode_id]
                    observations = ebuf[0]
                    obs_count = len(observations)
                    step_types = ebuf[5]
                    
                    # -> (1) 起始帧
                    if action == -860723 and obs_count==0:
                        #(1-1)
                        ebuf[0] = ts.observation
                        ebuf[5] = tf.concat((step_types,ts.step_type),axis=0)
                    elif step_types[-1].numpy() != 2:
                        # -> (2) 中间帧
                        ts_type_np = ts.step_type[0].numpy()
                        if ts_type_np == 2:
                            last_timesteps.add(episode_id)
                        limit_as_last = ts_type_np == 1 and obs_count==self.max_steps-1
                        discounts = ebuf[1]
                        rewards = ebuf[2]
                        n_step_types = ebuf[6]
                        st = tf.constant([2],dtype=tf.int32) if limit_as_last else ts.step_type
                        ebuf[0] = tf.concat((observations,ts.observation),axis=0)
                        ebuf[5] = tf.concat((step_types,st),axis=0)
                        ebuf[1] = tf.concat((discounts,ts.discount),axis=0)
                        ebuf[2] = tf.concat((rewards,ts.reward),axis=0)
                        ebuf[6] = tf.concat((n_step_types,st),axis=0)
                    else:
                        # -> (3) 结束帧
                        discounts = ebuf[1]
                        rewards = ebuf[2]
                        n_step_types = ebuf[6]
                        # 1. 填充episode_buffer(discounts,rewards,nex_step_type=first)
                        #print('current discounts len -> ', len(discounts))
                        #ebuf[1] = tf.concat((discounts,),axis=0)
                        #ebuf[2] = tf.concat((rewards,ts.reward),axis=0)
                        #ebuf[6] = tf.concat((n_step_types,tf.constant([0],dtype=tf.int32)),axis=0)
                        # 2. 利用episode_buffer中此episode的数据构建grpc调用结果yield发给调用端
                        actions = ebuf[4]
                        psinfos = ebuf[3]
                        episode = collector_pb2.Episode(
                            types=step_types.numpy().tolist(),
                            discounts=discounts.numpy().tolist()+ts.discount.numpy().tolist(),
                            rewards=rewards.numpy().tolist()+ts.reward.numpy().tolist(),
                            observations=tf.reshape(observations,(-1,)).numpy().tolist(),
                            next_types=n_step_types.numpy().tolist()+[0],
                            actions=actions.numpy().tolist(),
                            ps_infos=pickle.dumps(tf.nest.map_structure(lambda t:tf.expand_dims(t,axis=0),psinfos)),
                        )
                        num_collected+=1
                        yield episode
                        # 3. 清空episode_buffer中此episode数据,节约内存
                        ebuf.clear()
                        # 4. 查看num_to_collect是不是有新的episode需要收集，如果有，搬运118-132行同款代码
                        if num_to_collect > 0:
                            #开始一个episode，发送action=-860723
                            to_worker_queue.put((2, (episode_id_cursor, env_id, -860723)))
                            print('start a new collect task in for episode,worker,env -> ',episode_id_cursor, worker_id, env_id)
                            episode_id_cursor+=1
                            episode_buffer.append(
                                [[],#observations,0
                                tf.constant([],dtype=tf.float32),#discounts,1
                                tf.constant([],dtype=tf.float32),#rewards,2
                                (),#ps_infos,3
                                tf.constant([],dtype=tf.int64),#actions,4
                                tf.constant([],dtype=tf.int32),#step_types,5
                                tf.constant([],dtype=tf.int32),#next_step_types,6
                                self.collect_policy.get_initial_state(batch_size=1)#policy step states用来给collect_policy的action()使用,7
                                ])
                            num_to_collect -= 1
                        continue

                    if self.stateful:
                        policy_step_states.append(ebuf[7])
                    batched_epids.append(episode_id)
                    batched_tss.append(ts)
                    batched_worker_envs.append((worker_id,env_id))

                num_ts_collected = len(batched_tss)
                if num_ts_collected == 0:
                    continue

                #够一个batch了，执行action
                batched_ts = tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*batched_tss)
                #需要加入policy state
                ps_states = ()
                if self.stateful:
                    ps_states = tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*policy_step_states)
                    policy_step_states.clear()
                
                batched_ps = self.collect_policy.action(batched_ts, ps_states)
                #需要将policy step的相关信息加入到buffer中: action, ps_info, ps_state
                for posid in range(num_ts_collected):
                    worker_id, env_id = batched_worker_envs[posid]
                    to_worker_queue,_ = self.worker_queues[worker_id]
                    episode_id = batched_epids[posid]
                    ebuf = episode_buffer[episode_id]
                    ps_infos = ebuf[3]
                    actions = ebuf[4]
                    if batched_ps.info != ():
                        ep_ps_info = tf.nest.map_structure(lambda t:tf.expand_dims(t[posid],axis=0),batched_ps.info)
                        if ebuf[3] == ():
                            ebuf[3] = ep_ps_info
                        else:
                            ebuf[3] = tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*[ps_infos,ep_ps_info])
                    action_tf = batched_ps.action[posid]
                    ebuf[4] = tf.concat((actions,tf.expand_dims(action_tf,axis=0)),axis=0)
                    if batched_ps.state != ():
                        ebuf[7] = tf.nest.map_structure(lambda t:t[posid],batched_ps.state)
                
                    #向worker请求下一个step，发送(env_id, action)，注意当episode_id在last_timesteps里面时，action=-860723
                    to_worker_queue.put((2, (episode_id, env_id, -860723 if episode_id in last_timesteps else action_tf.numpy())))
                
                batched_epids.clear()
                last_timesteps.clear()
                batched_tss.clear()
                batched_worker_envs.clear()
        except Exception as e:
            print('An error occurred: ', e)
            traceback.print_exc()
            raise e
        
    def envspecs(self, request, context):
        return collector_pb2.EnvSpecResp(
            observation_spec=pickle.dumps(self.obs_spec),
            action_spec=pickle.dumps(self.act_spec),
            time_step_spec=pickle.dumps(self.ts_spec)
        )
    
def serve(num_worker, num_env, max_steps, env_creator, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = CollectorServicer(env_creator, num_worker, num_env, max_steps)
    collector_pb2_grpc.add_CollectServiceServicer_to_server(servicer, server)
    server.add_insecure_port('0.0.0.0:'+str(port))
    server.start()
    return server, servicer

if __name__ == '__main__':
    if args.num_worker:
        num_worker = args.num_worker
    else:
        num_worker = 2

    if args.num_env:
        num_env = args.num_env
    else:
        num_env = 1

    if args.max_steps:
        max_steps = args.max_steps
    else:
        max_steps = 100

    if args.port:
        port = args.port
    else:
        port =9555

    env_cfg = {}
    if args.env_creator:
        import functools
        import importlib
        parts = args.env_creator.split('.')
        if len(parts)>1:
            module = importlib.import_module(parts[0])
            e_creator = functools.reduce(getattr, [module]+parts[1:])
            if hasattr(module,'env_config'):
                env_cfg = getattr(module,'env_config').copy()
                env_cfg['total_partitions'] = num_env*num_worker
            env_creator = e_creator
        else:
            env_creator = eval(parts[0])
    else:
        def creator():
            env = suite_gym.load('CartPole-v0')
            return env
        env_creator = creator
    env_creator.config_template = env_cfg
    server, servicer = serve(num_worker, num_env, max_steps, env_creator, port)
    while True:
        try:
            inp = input("请输入policy ratio, 输入exit退出：")
            servicer.workload_p = float(inp)
        except:
            if inp == 'exit':
                os.kill(os.getpid(), signal.SIGTERM)
                break