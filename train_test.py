import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_logical_device_configuration(gpus[0], [
#     tf.config.LogicalDeviceConfiguration(memory_limit=9216)
# ])
addrs=["192.168.1.166:29550","192.168.1.166:29551","192.168.1.166:29552","192.168.1.166:29553","192.168.1.166:29554","192.168.1.166:29555"]

class LocalObserver:
    def __init__(self):
        self.trajs = []
    def observe(self, id, traj):
        self.trajs += [traj]

obs = LocalObserver()
from distribute_collector import DriverCollector
collector = DriverCollector(addrs,obs.observe,60)
import agent_create
counter,ppo=agent_create.ppo(collector.observation_spec, collector.action_spec, collector.time_step_spec)

collector.collect(120)
btrajs=tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*obs.trajs[:4])


ppo.train(obs.trajs[0])

btrajs=tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*obs.trajs)


