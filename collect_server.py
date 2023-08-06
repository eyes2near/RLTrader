# import os
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
# os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
# import tensorflow as tf
# import grpc
# import collector_pb2
# import collector_pb2_grpc
# from concurrent import futures
# import argparse
# from tf_agents.environments import suite_gym
# import traceback
# import pickle
# import agent_create
# from tf_agents.specs import tensor_spec
# from tf_agents.environments.tf_py_environment import TFPyEnvironment
# import numpy as np
# import time

# #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# parser = argparse.ArgumentParser()
# parser.add_argument("--num_env", type=int, help="number of envs to step")
# parser.add_argument("--max_steps", type=int, help="max steps env can take")
# parser.add_argument("--env_creator", type=str, help="env creator module and function")
# parser.add_argument("--port", type=str, help="port to start service")
# parser.add_argument("--partition_id", type=int, help="partition id to init env")
# args = parser.parse_args()

# class CollectorServicer(collector_pb2_grpc.CollectServiceServicer):

#     def __init__(self, env_creator, num_env, max_steps) -> None:
#         self.envs = []
#         #self.env_step_cusors = []
#         for _ in range(num_env):
#             env = env_creator()
#             if not isinstance(env, TFPyEnvironment):
#                 env = TFPyEnvironment(env)
#             self.envs.append(env)
#             #self.env_step_cusors.append(0)
#         self.max_steps = max_steps
#         obs_spec = tensor_spec.from_spec(self.envs[0].observation_spec())
#         action_spec = tensor_spec.from_spec(self.envs[0].action_spec())
#         ts_spec = tensor_spec.from_spec(self.envs[0].time_step_spec())
#         self.train_step_counter,agent = agent_create.ppo(obs_spec,action_spec,ts_spec)
#         self.collect_policy = agent.collect_policy
#         super().__init__()


#     def update_policy(self, request, context):
#         self.train_step_counter = pickle.loads(request.train_step_counter)
#         variables = pickle.loads(request.variables)
#         for variable, value in zip(
#             tf.nest.flatten(self.collect_policy.variables()), tf.nest.flatten(variables)):
#             variable.assign(value)
#         return collector_pb2.UpdatePolicyResp()
    
#     def collect(self, request, context):
#         try:
#             num_to_collect = request.num_episodes
#             num_collected = 0
#             while num_to_collect > 0:
#                 for env in self.envs:
#                     start = time.time()
#                     step_count = 1
#                     ts = env.reset()
#                     ts_types = np.array(0,dtype=np.int32)
#                     ts_observations = np.array(tf.reshape(ts.observation,(-1,)).numpy(),dtype=np.float32)
#                     ts_next_types = np.array([],dtype=np.int32)
#                     ts_discounts = np.array([],dtype=np.float32)
#                     ts_rewards = np.array([],dtype=np.float32)
#                     ps_actions = np.array([],dtype=np.int64)
#                     ps_state = self.collect_policy.get_initial_state(batch_size=1)
#                     ps_infos = []
#                     while not ts.is_last() and step_count < self.max_steps:
#                         step_count+=1
#                         act = self.collect_policy.action(ts,ps_state)
#                         ps_state = act.state
#                         if act.info != ():
#                             ps_infos.append(act.info)
#                         ps_actions = np.append(ps_actions,act.action.numpy())
#                         ts = env.step(act.action)
#                         ts_discounts = np.append(ts_discounts,ts.discount.numpy())
#                         ts_rewards = np.append(ts_rewards, ts.reward.numpy())
#                         ts_observations = np.append(ts_observations, tf.reshape(ts.observation,(-1,)).numpy())
#                         ts_types = np.append(ts_types,ts.step_type.numpy() if step_count != self.max_steps else 2)
#                         ts_next_types= np.append(ts_next_types,ts_types[-1])
#                     #向前继续step一步，如果ts.is_last()，则是reset一次
#                     final_act = self.collect_policy.action(ts,ps_state)
#                     if final_act.info != ():
#                         ps_infos.append(final_act.info)
#                     ps_actions = np.append(ps_actions,final_act.action.numpy())
#                     final_ts = env.reset() if ts.is_last() else env.step(final_act.action)
#                     ts_discounts = np.append(ts_discounts,final_ts.discount.numpy())
#                     ts_rewards = np.append(ts_rewards, final_ts.reward.numpy())
#                     ts_next_types= np.append(ts_next_types,0)
#                     psinfos = ()
#                     if len(ps_infos) > 0:
#                         psinfos = tf.nest.map_structure(lambda *tensors:tf.stack(tensors, axis=len(tensors[0].shape)-1), *ps_infos)
#                     episode = collector_pb2.Episode(
#                         types=ts_types.tolist(),
#                         discounts=ts_discounts.tolist(),
#                         rewards=ts_rewards.tolist(),
#                         observations=ts_observations.tolist(),
#                         next_types=ts_next_types.tolist(),
#                         actions=ps_actions.tolist(),
#                         ps_infos=pickle.dumps(psinfos),
#                     )
#                     num_to_collect-=1
#                     num_collected+=1
#                     print('num episodes collected -> ', num_collected, 'time spent -> ', time.time()-start, ' seconds')
#                     yield episode
#         except Exception as e:
#             print('An error occurred: ', e)
#             traceback.print_exc()
#             raise e
        
#     def envspecs(self, request, context):
#         env = self.envs[0]
#         return collector_pb2.EnvSpecResp(
#             observation_spec=pickle.dumps(env.observation_spec()),
#             action_spec=pickle.dumps(env.action_spec()),
#             time_step_spec=pickle.dumps(env.time_step_spec())
#         )
    
# def serve(num_env, max_steps, env_creator, port):
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
#     collector_pb2_grpc.add_CollectServiceServicer_to_server(CollectorServicer(env_creator, num_env, max_steps), server)
#     server.add_insecure_port('localhost:'+str(port))
#     server.start()
#     server.wait_for_termination()

# if __name__ == '__main__':
#     if args.num_env:
#         num_env = args.num_env
#     else:
#         num_env = 1

#     if args.max_steps:
#         max_steps = args.max_steps
#     else:
#         max_steps = 100

#     if args.port:
#         port = args.port
#     else:
#         port =9555

#     if args.partition_id:
#         partition_id = args.partition_id
#     else:
#         partition_id = 0

#     if args.env_creator:
#         import functools
#         import importlib
#         parts = args.env_creator.split('.')
#         if len(parts)>1:
#             module = importlib.import_module(parts[0])
#             e_creator = functools.reduce(getattr, [module]+parts[1:])
#             if hasattr(module,'env_config'):
#                 env_config = getattr(module,'env_config').copy()
#                 env_config['partition_id'] = partition_id
#                 env_creator = lambda : e_creator(env_config)
#         else:
#             env_creator = eval(parts[0])
#     else:
#         def creator():
#             env = suite_gym.load('CartPole-v0')
#             return env
#         env_creator = creator
#     serve(num_env, max_steps, env_creator, port)
    