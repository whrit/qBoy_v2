import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import keras.backend as K
import gymnasium as gym
from src.replay import PrioritizedReplayBuffer

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    device = '/GPU:0'
    print("Using GPU for training")
else:
    device = '/CPU:0'
    print("Using CPU for training")

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

class DoubleQLearningAgent:
    def __init__(self, env, learning_rate, discount_factor, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, bins=10, buffer_size=10000, batch_size=32):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table1 = {}
        self.q_table2 = {}
        self.bins = bins
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=0.6)
        self.beta = 0.4

        if isinstance(env.single_action_space, gym.spaces.Discrete):
            self.action_space_type = 'Discrete'
            self.action_space_size = env.single_action_space.n
        elif isinstance(env.single_action_space, gym.spaces.MultiDiscrete):
            self.action_space_type = 'MultiDiscrete'
            self.action_space_size = env.single_action_space.nvec
        else:
            raise ValueError("Unsupported action space type")

    def discretize_state(self, state):
        discretized_state = tuple((state * self.bins).astype(int))
        return discretized_state

    def step(self, state):
        observation = state["observation"]
        state_index = self.discretize_state(observation)
        if np.random.rand() < self.epsilon:
            if self.action_space_type == 'Discrete':
                action = np.random.randint(self.action_space_size)
            elif self.action_space_type == 'MultiDiscrete':
                action = [np.random.randint(n) for n in self.action_space_size]
        else:
            q_values1 = self.q_table1.get(state_index, np.zeros(self.action_space_size))
            q_values2 = self.q_table2.get(state_index, np.zeros(self.action_space_size))
            q_values = q_values1 + q_values2
            if self.action_space_type == 'Discrete':
                action = np.argmax(q_values)
            elif self.action_space_type == 'MultiDiscrete':
                action = [np.argmax(q) for q in q_values]
        
        return action

    def update_qvalue(self, state, action, reward, next_state, done):
        observation = state["observation"]
        next_observation = next_state["observation"]
        state_index = self.discretize_state(observation)
        next_state_index = self.discretize_state(next_observation)

        current_q = (self.q_table1.get(state_index, np.zeros(self.action_space_size))[action] + 
                     self.q_table2.get(state_index, np.zeros(self.action_space_size))[action]) / 2

        if done:
            target = reward
        else:
            if self.action_space_type == 'Discrete':
                best_next_action = np.argmax(self.q_table1.get(next_state_index, np.zeros(self.action_space_size)) +
                                             self.q_table2.get(next_state_index, np.zeros(self.action_space_size)))
                target = reward + self.discount_factor * (self.q_table1.get(next_state_index, np.zeros(self.action_space_size))[best_next_action] +
                                                          self.q_table2.get(next_state_index, np.zeros(self.action_space_size))[best_next_action]) / 2
            elif self.action_space_type == 'MultiDiscrete':
                best_next_action = [np.argmax(q1 + q2) for q1, q2 in zip(self.q_table1.get(next_state_index, np.zeros(self.action_space_size)),
                                                                         self.q_table2.get(next_state_index, np.zeros(self.action_space_size)))]
                target = reward + self.discount_factor * sum((self.q_table1.get(next_state_index, np.zeros(self.action_space_size))[i, best_next_action[i]] +
                                                              self.q_table2.get(next_state_index, np.zeros(self.action_space_size))[i, best_next_action[i]]) / 2
                                                             for i in range(len(best_next_action)))

        td_error = abs(target - current_q)
        self.memory.add(td_error, (state, action, reward, next_state, done))

        if np.random.rand() < 0.5:
            self.q_table1[state_index] = self.q_table1.get(state_index, np.zeros(self.action_space_size))
            self.q_table1[state_index][action] += self.learning_rate * (target - self.q_table1[state_index][action])
        else:
            self.q_table2[state_index] = self.q_table2.get(state_index, np.zeros(self.action_space_size))
            self.q_table2[state_index][action] += self.learning_rate * (target - self.q_table2[state_index][action])

    def replay(self):
        if len(self.memory.tree.data) < self.batch_size:
            return

        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            state_index = self.discretize_state(state["observation"])
            next_state_index = self.discretize_state(next_state["observation"])
            
            current_q = (self.q_table1.get(state_index, np.zeros(self.action_space_size))[action] + 
                         self.q_table2.get(state_index, np.zeros(self.action_space_size))[action]) / 2

            if done:
                target = reward
            else:
                if self.action_space_type == 'Discrete':
                    best_next_action = np.argmax(self.q_table1.get(next_state_index, np.zeros(self.action_space_size)) +
                                                 self.q_table2.get(next_state_index, np.zeros(self.action_space_size)))
                    target = reward + self.discount_factor * (self.q_table1.get(next_state_index, np.zeros(self.action_space_size))[best_next_action] +
                                                              self.q_table2.get(next_state_index, np.zeros(self.action_space_size))[best_next_action]) / 2
                elif self.action_space_type == 'MultiDiscrete':
                    best_next_action = [np.argmax(q1 + q2) for q1, q2 in zip(self.q_table1.get(next_state_index, np.zeros(self.action_space_size)),
                                                                             self.q_table2.get(next_state_index, np.zeros(self.action_space_size)))]
                    target = reward + self.discount_factor * sum((self.q_table1.get(next_state_index, np.zeros(self.action_space_size))[i, best_next_action[i]] +
                                                                  self.q_table2.get(next_state_index, np.zeros(self.action_space_size))[i, best_next_action[i]]) / 2
                                                                 for i in range(len(best_next_action)))

            td_error = abs(target - current_q)
            self.memory.update(idxs[i], td_error)

            if np.random.rand() < 0.5:
                self.q_table1[state_index] = self.q_table1.get(state_index, np.zeros(self.action_space_size))
                self.q_table1[state_index][action] += self.learning_rate * (target - self.q_table1[state_index][action]) * is_weights[i]
            else:
                self.q_table2[state_index] = self.q_table2.get(state_index, np.zeros(self.action_space_size))
                self.q_table2[state_index][action] += self.learning_rate * (target - self.q_table2[state_index][action]) * is_weights[i]

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN Agent
class DQNAgent:
    def __init__(self, env, model_name='DQN_model', gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.env = env
        self.model_name = model_name
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.buffer_size = 10000
        self.batch_size = 32
        self.memory = PrioritizedReplayBuffer(self.buffer_size, 0.6)
        self.beta = 0.4
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.env.observation_space.shape[1]))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.env.action_space.n))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        state = state.flatten().reshape(1, -1)
        next_state = next_state.flatten().reshape(1, -1)
        target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]) if not done else reward
        td_error = abs(target - np.amax(self.model.predict(state)[0]))
        self.memory.add(td_error, (state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.env.action_space.n)
        state = state.flatten().reshape(1, -1)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory.tree.data) < self.batch_size:
            return
        minibatch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        targets = self.model.predict(states)
        targets_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            target = rewards[i] + self.gamma * np.amax(targets_next[i]) if not dones[i] else rewards[i]
            td_error = abs(target - targets[i][actions[i]])
            self.memory.update(idxs[i], td_error)
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)
        self.update_target_model()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay