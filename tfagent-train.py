from multiprocess_collector import Collectors
import env_create
from tf_agents.environments import suite_gym
from tf_agents.networks import actor_distribution_network
import tensorflow as tf
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import tf_py_environment
import time
from tf_agents.replay_buffers.episodic_replay_buffer import EpisodicReplayBuffer

learning_rate=1e-3

collector_service_addrs=['localhost:10955','localhost:10956','localhost:10957',
                         'localhost:10958','localhost:10959','localhost:10960',]

num_envs_per_collector=5

max_collect_steps_per_episode=100

def train():
    pass

if __name__ == '__main__':
    train()