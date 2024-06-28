import numpy as np
import random
from collections import deque

class QLearningAgent:
    def __init__(self, env, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon, buffer_size, batch_size):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def step(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space.n)
        return np.argmax(self.q_table[state,:])

    def update_qvalue(self, current_state, current_action, reward, future_state):
        best_future_q = np.max(self.q_table[future_state,:])
        self.q_table[current_state, current_action] += self.learning_rate * (
            reward + self.discount_factor * best_future_q - self.q_table[current_state, current_action]
        )

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
            self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.step(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_qvalue(state, action, reward, next_state)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                self.experience_replay()
            self.decay_epsilon()