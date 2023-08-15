import tensorflow as tf
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import value_network
import copy
import time

class AgentConfig:
    def copy(self):
        return copy.copy(self)

DEFAULT_RF_AGENT_CONFIG = AgentConfig()
DEFAULT_RF_AGENT_CONFIG.lr = 1e-3
DEFAULT_RF_AGENT_CONFIG.fc_layer_params=(512,128)

DEFAULT_PPO_AGENT_CONFIG = AgentConfig()
# Training params
DEFAULT_PPO_AGENT_CONFIG.num_iterations=1600
DEFAULT_PPO_AGENT_CONFIG.actor_fc_layers=(512, 128)
DEFAULT_PPO_AGENT_CONFIG.value_fc_layers=(512, 128)
DEFAULT_PPO_AGENT_CONFIG.learning_rate=3e-4
DEFAULT_PPO_AGENT_CONFIG.collect_sequence_length=2048
DEFAULT_PPO_AGENT_CONFIG.minibatch_size=64
DEFAULT_PPO_AGENT_CONFIG.num_epochs=10
# Agent params
DEFAULT_PPO_AGENT_CONFIG.importance_ratio_clipping=0.2
DEFAULT_PPO_AGENT_CONFIG.lambda_value=0.95
DEFAULT_PPO_AGENT_CONFIG.discount_factor=0.99
DEFAULT_PPO_AGENT_CONFIG.entropy_regularization=0.
DEFAULT_PPO_AGENT_CONFIG.value_pred_loss_coef=0.5
DEFAULT_PPO_AGENT_CONFIG.use_gae=True,
DEFAULT_PPO_AGENT_CONFIG.use_td_lambda_return=True
DEFAULT_PPO_AGENT_CONFIG.gradient_clipping=0.5
DEFAULT_PPO_AGENT_CONFIG.value_clipping=None
DEFAULT_PPO_AGENT_CONFIG.debug_summaries=False
DEFAULT_PPO_AGENT_CONFIG.summarize_grads_and_vars=False
DEFAULT_PPO_AGENT_CONFIG.compute_value_and_advantage_in_train=True
DEFAULT_PPO_AGENT_CONFIG.update_normalizers_in_train=False

# def rf_agent(config=DEFAULT_RF_AGENT_CONFIG):
#     learning_rate=config.lr
#     fc_layer_params = config.fc_layer_params
#     train_step_counter = tf.Variable(0)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     actor_net = actor_distribution_network.ActorDistributionNetwork(
#         config.observation_spec,
#         config.action_spec,
#         fc_layer_params=fc_layer_params)
#     agent = reinforce_agent.ReinforceAgent(
#         config.time_step_spec,
#         config.action_spec,
#         actor_network=actor_net,
#         optimizer=optimizer,
#         normalize_returns=True,
#         train_step_counter=train_step_counter)
#     agent.initialize()
#     return (train_step_counter, agent)

def ppo_agent(config=DEFAULT_PPO_AGENT_CONFIG):
    preprocessing_layers = {
            "market":createActorAttensionLayers(),
            "stateful":tf.keras.models.Sequential([
                tf.keras.layers.Dense(64,activation="relu"), 
                tf.keras.layers.Dense(16,activation="relu"),
                tf.keras.layers.Dropout(0.2)
                ])
        }
    preprocessing_combiner = createObservationPreprocessCombiner()
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        config.observation_spec,
        config.action_spec,
        preprocessing_layers=preprocessing_layers,
      preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=config.actor_fc_layers)
    
    value_net = value_network.ValueNetwork(
      config.observation_spec,
      preprocessing_layers=preprocessing_layers,
      preprocessing_combiner=preprocessing_combiner,
      fc_layer_params=config.value_fc_layers,
      kernel_initializer=tf.keras.initializers.Orthogonal(seed=int(time.time())))

    train_step_counter = tf.Variable(0)
    
    agent = ppo_clip_agent.PPOClipAgent(
        config.time_step_spec,
        config.action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate_fn, epsilon=1e-5),
        actor_net=actor_net,
        value_net=value_net,
        importance_ratio_clipping=config.importance_ratio_clipping,
        lambda_value=config.lambda_value,
        discount_factor=config.discount_factor,
        entropy_regularization=config.entropy_regularization,
        value_pred_loss_coef=config.value_pred_loss_coef,
        # This is a legacy argument for the number of times we repeat the data
        # inside of the train function, incompatible with mini batch learning.
        # We set the epoch number from the replay buffer and tf.Data instead.
        num_epochs=1,
        use_gae=config.use_gae,
        use_td_lambda_return=config.use_td_lambda_return,
        gradient_clipping=config.gradient_clipping,
        value_clipping=config.value_clipping,
        compute_value_and_advantage_in_train=config.compute_value_and_advantage_in_train,
        # Skips updating normalizers in the agent, as it's handled in the learner.
        update_normalizers_in_train=config.update_normalizers_in_train,
        debug_summaries=config.debug_summaries,
        summarize_grads_and_vars=config.summarize_grads_and_vars,
        train_step_counter=train_step_counter)
    agent.initialize()
    return (train_step_counter, agent)

def ppo(obs_spec,action_spec,ts_spec):
    config = DEFAULT_PPO_AGENT_CONFIG.copy()
    config.observation_spec=obs_spec
    config.action_spec=action_spec
    config.time_step_spec=ts_spec
    config.learning_rate_fn=lambda:config.learning_rate
    return ppo_agent(config)

# def rf(obs_spec,action_spec,ts_spec):
#     config = DEFAULT_RF_AGENT_CONFIG.copy()
#     config.observation_spec=obs_spec
#     config.action_spec=action_spec
#     config.time_step_spec=ts_spec
#     return rf_agent(config)

def createActorAttensionLayers(): 
    return TransformerLayer(hidden_units=64, num_layers=4, num_attention_heads=6, dropout_rate=0.1)

def createObservationPreprocessCombiner():
    return tf.keras.layers.Concatenate(axis=-1)
    
class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, num_layers, num_attention_heads, dropout_rate, **kwargs):
        super(TransformerLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Define any necessary variables here, if needed
        super(TransformerLayer, self).build(input_shape)

    def call(self, inputs):
        hidden_units = self.hidden_units
        num_layers = self.num_layers
        num_attention_heads = self.num_attention_heads
        dropout_rate = self.dropout_rate

        prev_output = inputs
        res_output = prev_output

        # Create a layer that includes the attention mechanism for preprocessing,
        # actor_net and value_net will share this layer
        for _ in range(num_layers):
            # Add residual connection and layer normalization
            prev_output = res_output  # Save the previous layer's output for residual connection
            res_output = tf.keras.layers.MultiHeadAttention(
                key_dim=hidden_units // num_attention_heads, num_heads=num_attention_heads, dropout=dropout_rate,
            )(res_output, res_output)  # Attention layer
            res_output = tf.keras.layers.Dropout(dropout_rate)(res_output)  # Add Dropout layer
            res_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res_output + prev_output)  # Residual connection and Layer normalization
            res_output = tf.keras.layers.Dense(hidden_units, activation='relu')(res_output)  # Add feed-forward fully connected layer
            res_output = tf.keras.layers.Dropout(dropout_rate)(res_output)  # Add Dropout layer
            res_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res_output)  # Layer normalization

        # Define the output layer
        res_output = tf.keras.layers.Flatten()(res_output)  # Output layer

        return res_output
   
    def get_config(self):
        config = super(TransformerLayer, self).get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config