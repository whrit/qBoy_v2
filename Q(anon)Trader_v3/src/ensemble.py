import tensorflow as tf
import logging
from src.agent import DoubleQLearningAgent

tf.get_logger().setLevel(logging.ERROR)

class AgentEnsemble:
    def __init__(self, num_agents, observation_space, action_space, learning_rate, discount_factor, **kwargs):
        self.agents = [DoubleQLearningAgent(observation_space, action_space, learning_rate, discount_factor, **kwargs) for _ in range(num_agents)]

    def act(self, state):
        actions = [agent.step(state) for agent in self.agents]
        return max(set(actions), key=actions.count)  # majority vote

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),  # Assuming states have shape [batch_size, 2]
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),  # Assuming next_states have shape [batch_size, 2]
        tf.TensorSpec(shape=[None], dtype=tf.bool)
    ])
    def update(self, states, actions, rewards, next_states, dones):
        td_errors = []
        for agent in self.agents:
            td_error = agent.update_qvalue_batch(states, actions, rewards, next_states, dones)
            td_errors.append(td_error)
        return states, actions, rewards, next_states, dones, tf.stack(td_errors)

    def update_memories(self, states, actions, rewards, next_states, dones, td_errors):
        for agent, td_error in zip(self.agents, td_errors):
            agent.update_memory(states, actions, rewards, next_states, dones, td_error)

    def decay_epsilon(self):
        for agent in self.agents:
            agent.decay_epsilon()

    def replay(self):
        for agent in self.agents:
            agent.replay()