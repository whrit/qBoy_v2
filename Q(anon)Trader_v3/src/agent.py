import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K
import gymnasium as gym
from src.replay import PrioritizedReplayBuffer
import logging

tf.get_logger().setLevel(logging.ERROR)

tf.config.run_functions_eagerly(True)

@tf.function
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

class DoubleQLearningAgent:
    def __init__(self, observation_space, action_space, learning_rate, discount_factor, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=32):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = PrioritizedReplayBuffer(self.buffer_size, alpha=0.6)
        self.beta = 0.4

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_space_type = 'Discrete'
            self.action_space_size = action_space.n
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            self.action_space_type = 'MultiDiscrete'
            self.action_space_size = action_space.nvec
        else:
            raise ValueError("Unsupported action space type")

        self.model1 = self._build_model()
        self.model2 = self._build_model()

    def _build_model(self):
        input_shape = self.observation_space.shape
        inputs = Input(shape=input_shape)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(np.sum(self.action_space_size) if self.action_space_type == 'MultiDiscrete' else self.action_space_size, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=huber_loss, optimizer=Adam(learning_rate=self.learning_rate))
        return model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 2], dtype=tf.float32),  # Assuming states have shape [batch_size, 2]
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),  # Assuming target has shape [batch_size, 3]
        tf.TensorSpec(shape=[], dtype=tf.int32)  # model_idx as an integer
    ])
    def _apply_gradients(self, states, target, model_idx):
        def update_model1():
            with tf.GradientTape() as tape:
                q_values = self.model1(states)
                loss = huber_loss(target, q_values)
                gradients = tape.gradient(loss, self.model1.trainable_variables)
                self.model1.optimizer.apply_gradients(zip(gradients, self.model1.trainable_variables))
            return loss

        def update_model2():
            with tf.GradientTape() as tape:
                q_values = self.model2(states)
                loss = huber_loss(target, q_values)
                gradients = tape.gradient(loss, self.model2.trainable_variables)
                self.model2.optimizer.apply_gradients(zip(gradients, self.model2.trainable_variables))
            return loss

        return tf.cond(tf.equal(model_idx, 1), update_model1, update_model2)

    def _reshape_input(self, state):
        if len(state.shape) == 1:
            return np.expand_dims(state, axis=0)
        return state

    def step(self, state):
        if np.random.rand() < self.epsilon:
            if self.action_space_type == 'Discrete':
                return np.random.randint(self.action_space_size)
            elif self.action_space_type == 'MultiDiscrete':
                return np.array([np.random.randint(n) for n in self.action_space_size])
        else:
            state = self._reshape_input(np.array(state["observation"]))
            q_values1 = self.model1.predict(state, verbose=0)
            q_values2 = self.model2.predict(state, verbose=0)
            q_values = q_values1 + q_values2
            if self.action_space_type == 'Discrete':
                return np.argmax(q_values[0])
            elif self.action_space_type == 'MultiDiscrete':
                actions = []
                start = 0
                for n in self.action_space_size:
                    actions.append(np.argmax(q_values[0][start:start+n]))
                    start += n
                return np.array(actions)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.bool)
    ])
    def update_qvalue_batch(self, states, actions, rewards, next_states, dones):
        future_returns1 = self.model1(next_states)
        future_returns2 = self.model2(next_states)
        
        if self.action_space_type == 'Discrete':
            next_q_values = tf.reduce_max(future_returns1, axis=1)
        elif self.action_space_type == 'MultiDiscrete':
            next_q_values = tf.zeros_like(rewards)
            start = 0
            for n in self.action_space_size:
                next_q_values += tf.reduce_max(future_returns1[:, start:start+n], axis=1)
                start += n
            next_q_values /= len(self.action_space_size)

        targets = rewards + self.discount_factor * next_q_values * (1 - tf.cast(dones, tf.float32))
        
        target_f1 = self.model1(states)
        target_f2 = self.model2(states)

        if self.action_space_type == 'Discrete':
            indices = tf.stack([tf.range(tf.shape(actions)[0]), tf.cast(actions, tf.int32)], axis=1)
            target_f1 = tf.tensor_scatter_nd_update(target_f1, indices, targets)
            target_f2 = tf.tensor_scatter_nd_update(target_f2, indices, targets)
        elif self.action_space_type == 'MultiDiscrete':
            start = 0
            for i, n in enumerate(self.action_space_size):
                indices = tf.stack([tf.range(tf.shape(actions)[0]), start + tf.cast(actions[:, i], tf.int32)], axis=1)
                target_f1 = tf.tensor_scatter_nd_update(target_f1, indices, targets)
                target_f2 = tf.tensor_scatter_nd_update(target_f2, indices, targets)
                start += n

        loss1 = self._apply_gradients(states, target_f1, tf.constant(1, dtype=tf.int32))
        loss2 = self._apply_gradients(states, target_f2, tf.constant(2, dtype=tf.int32))

        return tf.maximum(loss1, loss2)

    def replay(self):
        if len(self.memory.tree.data) < self.batch_size:
            return

        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        
        states = np.array([item[0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3] for item in batch])
        dones = np.array([item[4] for item in batch])

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)  # Changed to bool
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)

        td_errors = self.update_qvalue_batch(states, actions, rewards, next_states, dones)
        self.update_memory(states, actions, rewards, next_states, dones, td_errors)

    def update_memory(self, states, actions, rewards, next_states, dones, td_error):
        states = states.numpy()
        actions = actions.numpy()
        rewards = rewards.numpy()
        next_states = next_states.numpy()
        dones = dones.numpy()
        td_error = td_error.numpy()

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(td_error, (state, action, reward, next_state, done))

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Disable eager execution after debugging
tf.config.run_functions_eagerly(False)

# DQN Agent remains the same
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
        model.add(Dense(units=128, activation="relu", input_dim=self.env.observation_space.shape[0]))
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