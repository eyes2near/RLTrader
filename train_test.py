import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
# os.environ["TF_GPU_ALLOCATOR"]='cuda_malloc_async'
import env_create
import agent_create
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [
            tf.config.LogicalDeviceConfiguration(memory_limit=5120)
        ])
env=env_create.train_env()
counter,ppo=agent_create.ppo(env.tf.observation_spec(), env.tf.action_spec(), env.tf.time_step_spec())
addrs=["localhost:15515"]

class LocalObserver:
    def __init__(self):
        self.trajs = []
    def observe(self, id, traj):
        self.trajs += [traj]


obs = LocalObserver()

from distribute_collector import DriverCollector
collector = DriverCollector(addrs,obs.observe,20)
collector.collect(3)
btrajs=tf.nest.map_structure(lambda *t:tf.concat(t,axis=0),*obs.trajs)