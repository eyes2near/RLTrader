# import tensorflow as tf
# import time
# import multiprocessing
# import threading
# from tf_agents.trajectories import time_step as ts
# from tf_agents.trajectories.time_step import TimeStep, StepType
# from tf_agents.replay_buffers.episodic_replay_buffer import EpisodicReplayBuffer
# from tf_agents.trajectories import trajectory
# from tf_agents.trajectories.policy_step import PolicyStep

# def collect_process(conn, env_creator, num_envs=1, max_steps=10):
#     envs = []
#     for _ in range(num_envs):
#         env = env_creator()
#         envs.append(env)
#     step_counts = [0] * num_envs
#     while True:
#         #print('Begin to collect recv.')
#         env_ids, actions = conn.recv()
#         #print('Recved, ', env_ids, actions)
#         if isinstance(actions, str):
#             if actions == 'TERMINATE':
#                 break
#         start = time.time()
#         #将timesteps转化成一个batched timestep
#         step_types = []
#         rewards = []
#         discounts = []
#         observations = []
#         for ei in env_ids:
#             env = envs[ei]
#             action = actions[ei]
#             step_counts[ei]+=1
#             if action == None:
#                 #需要reset
#                 timestep = env.reset()
#                 step_types.append(timestep.step_type)
#                 rewards.append(timestep.reward)
#                 discounts.append(timestep.discount)
#                 observations.append(timestep.observation)
#                 step_counts[ei]=1
#                 continue
#             timestep = env.step(action) 
#             if step_counts[ei] == max_steps:
#                 timestep = TimeStep(tf.constant([StepType.LAST]),timestep.reward,timestep.discount,timestep.observation)
#                 step_counts[ei]=0
#             elif timestep.is_last():
#                 step_counts[ei]=0
#             step_types.append(timestep.step_type)
#             rewards.append(timestep.reward)
#             discounts.append(timestep.discount)
#             observations.append(timestep.observation)
        
#         if len(env_ids)>1:
#             timestep = TimeStep(tf.stack(step_types),
#                             tf.stack(rewards),
#                             tf.stack(discounts),
#                             tf.stack(observations))
#         conn.send((env_ids,timestep))
#         end = time.time()
#         elapsed_time = end - start
#         print(f"Envs step takes: {elapsed_time:.2f} seconds")

# class Collectors:
#     def __init__(self, env_creator, num_collectors, num_envs_per_collector, max_steps) -> None:
#         self.env_creator = env_creator
#         self.num_collectors = num_collectors
#         self.num_envs_per_collector = num_envs_per_collector
#         self.max_steps = max_steps
#         self.collector_conns = []
#         self.collector_processes = []
#         self.is_stop = False
#         self.is_terminate = False
    
#     def stop_collect(self):
#         self.is_stop = True
#         if self.collect_thread and self.collect_thread.is_alive():
#             try:
#                 self.collect_thread.join()
#             except:
#                 pass
#         self.is_stop=False

#     def terminate(self):
#         self.is_stop = True
#         self.is_terminate = True
#         if hasattr(self,'collect_thread') and self.collect_thread.is_alive():
#             try:
#                 self.collect_thread.join()
#             except:
#                 pass
#         for i in range(self.num_collectors):
#             conn = self.collector_conns[i]
#             process = self.collector_processes[i]
#             conn.send((None,"TERMINATE"))
#             conn.close()
#             process.join()
        
#     def init_collectors(self):
#         if len(self.collector_conns) > 0:
#             return
#         for _ in range(self.num_collectors):
#             conn, cconn = multiprocessing.Pipe()
#             p = multiprocessing.Process(target=collect_process, args=(cconn, self.env_creator, self.num_envs_per_collector, self.max_steps,))
#             p.start()
#             self.collector_processes.append(p)
#             self.collector_conns.append(conn)

#     def collect(self, collect_policy, replay_buffer:EpisodicReplayBuffer, num_episodes, async_collect=False):
#         if self.is_terminate:
#             return
#         if len(self.collector_conns) == 0:
#             self.init_collectors()

#         def do_collect():
#             #time.sleep(100)
#             if self.is_stop:
#                 return
#             collector_episode_ids = []
#             collector_current_steps = []
#             collector_policy_states = []
#             replay_buffer.clear()
#             #先看看一共需要收集多少个episode，然后分配给collector，每个collector可以一批处理num_envs_per_collector个episode
#             #需要保持最大的并行度,所以先num_episode/num_collectors,看看每个collector可以分配到多少个episode要处理
#             #然后再每一个collector中创建这些episode,并开始step
#             episode_collected = min(num_episodes,self.num_collectors*self.num_envs_per_collector)
#             num_e_in_collector = min(num_episodes//self.num_collectors,self.num_envs_per_collector)
#             num_episodes_in_collector = []
#             collector_done_flags = []
#             for ci in range(self.num_collectors):
#                 conn = self.collector_conns[ci]
#                 collector_current_steps.append([None]*self.num_envs_per_collector)
#                 num_episodes_in_collector.append(num_e_in_collector)
#                 if num_e_in_collector < self.num_envs_per_collector and ci < num_episodes%self.num_collectors:
#                     num_episodes_in_collector[ci]+=1
#                 collector_done_flags.append(num_episodes_in_collector[ci] == 0)
#                 episode_ids = None if collector_done_flags[ci] else [tf.constant([-1],dtype=tf.int64)]*num_episodes_in_collector[ci]
#                 collector_policy_states.append(collect_policy.get_initial_state(batch_size=num_episodes_in_collector[ci]))
#                 collector_episode_ids.append(episode_ids)
#                 #发送第一批step请求，使用None作为action，用来触发env的reset
#                 conn.send(([i for i in range(num_episodes_in_collector[ci])],[None]*num_episodes_in_collector[ci]))
#             time.sleep(100)
#             while not self.is_stop:
#                 all_done = True
#                 for ci in range(self.num_collectors):
#                     done = collector_done_flags[ci]
#                     if done:
#                         continue
#                     episode_ids = collector_episode_ids[ci]
#                     policy_states = collector_policy_states[ci]
#                     conn = self.collector_conns[ci]
#                     current_steps = collector_current_steps[ci]
#                     env_ids, timestep = conn.recv() #数据为:(env_ids,timesteps)
#                     print('action on steps, ', timestep)
#                     print('###########################################')
#                     action_steps = collect_policy.action(timestep, policy_states)
#                     policy_states=action_steps.state
#                     actions = []
#                     env_ids_to_send = []
#                     if len(env_ids) > 1:
#                         action_it = iter(action_steps.action)
#                         ts_type_it = iter(timestep.step_type)
#                         ts_reward_it = iter(timestep.reward)
#                         ts_discount_it = iter(timestep.discount)
#                         ts_obs_it = iter(timestep.observation)
#                         if policy_states:
#                             idx_it = iter(range(len(env_ids)))
#                         for env_id in env_ids:
#                             action = PolicyStep(action_it.next(),action_steps.info,action_steps.state)
#                             ts_type = ts_type_it.next()
#                             ts_reward = ts_reward_it.next()
#                             ts_discount = ts_discount_it.next()
#                             ts_obs = ts_obs_it.next()
#                             next_ts = TimeStep(ts_type,ts_reward,ts_discount,ts_obs)
#                             if policy_states:
#                                 idx = next(idx_it)
#                             if ts_type == StepType.FIRST:
#                                 #如果是起始帧，则不建立traj，直接发送数据继续step
#                                 actions.append(action)
#                                 env_ids_to_send.append(env_id)
#                                 current_steps[env_id]=next_ts
#                             elif ts_type == StepType.MID:
#                                 #如果是中间帧，建立traj，并且发送数据继续收集下一帧
#                                 new_id = replay_buffer.add_batch(trajectory.from_transition(current_steps[env_id], action, next_ts),episode_ids[env_id])
#                                 episode_ids[env_id]=new_id
#                                 actions.append(action)
#                                 env_ids_to_send.append(env_id)
#                                 current_steps[env_id]=next_ts
#                             else:
#                                 #结尾帧，此时代表已经收集完成一个episode
#                                 episode_collected+=1
#                                 traj = trajectory.from_transition(current_steps[env_id], action, next_ts)
#                                 new_id = replay_buffer.add_batch(traj,episode_ids[env_id])
#                                 episode_ids[env_id]=tf.constant([-1],dtype=tf.int64)
#                                 current_steps[env_id]=None
#                                 #如果已经采集够了，则不应继续在此env采集了
#                                 #否则继续使用此env收集下一个episode
#                                 if episode_collected <= num_episodes:
#                                     if policy_states:
#                                         policy_states[idx]=collect_policy.get_initial_state(batch_size=1)
#                                     env_ids_to_send.append(env_id)
#                                     #发送None作为action，用以通知env进行reset
#                                     actions.append(None)
#                                 else:
#                                     policy_states = policy_states[:idx]+policy_states[idx+1:]
#                     else:
#                         env_id = env_ids[0]
#                         #剩下单个的timestep了
#                         if timestep.is_first():
#                             #如果是起始帧，则不建立traj，直接发送数据继续step
#                             actions.append(action_steps.action)
#                             env_ids_to_send.append(env_id)
#                             current_steps[env_id]=timestep
#                         elif timestep.is_mid():
#                             #如果是中间帧，建立traj，并且发送数据继续收集下一帧
#                             new_id = replay_buffer.add_batch(trajectory.from_transition(current_steps[env_id], action_steps, timestep),episode_ids[env_id])
#                             episode_ids[env_id]=new_id
#                             actions.append(action_steps.action)
#                             env_ids_to_send.append(env_id)
#                             current_steps[env_id]=timestep
#                         else:
#                             #结尾帧，此时代表已经收集完成一个episode
#                             episode_collected+=1
#                             #建立traj
#                             traj = trajectory.from_transition(current_steps[env_id], action_steps, timestep)
#                             new_id = replay_buffer.add_batch(traj,episode_ids[env_id])
#                             episode_ids[env_id]=tf.constant([-1],dtype=tf.int64)
#                             current_steps[env_id]=None
#                             #如果已经采集够了，则不应继续在此env采集了
#                             #否则继续使用此env收集下一个episode
#                             if episode_collected <= num_episodes:
#                                 if policy_states:
#                                     policy_states[idx]=collect_policy.get_initial_state(batch_size=1)
#                                 env_ids_to_send.append(env_id)
#                                 #发送None作为action，用以通知env进行reset
#                                 actions.append(None)
#                             #else:
#                                 #policy_states = policy_states[:idx]+policy_states[idx+1:]
#                     if len(env_ids_to_send) > 0:
#                         conn.send((env_ids_to_send, actions))
#                         all_done = False
#                     else:
#                         #否则说明此collector收集完成了，不因该再继续尝试recv这个collector的conn了
#                         collector_done_flags[ci] = True
#                 if all_done:
#                     break
#             #采集完成或者被stop，需要抽干pipe
#             for conn in self.collector_conns:
#                 try:
#                     while conn.poll():
#                         conn.recv()
#                 except:
#                     continue
#             self.is_stop = False
        
#         if async_collect:
#             self.collect_thread = threading.Thread(target=do_collect)
#             self.collect_thread.start()
#         else:
#             do_collect()