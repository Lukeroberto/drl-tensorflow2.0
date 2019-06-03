import random
from collections import deque
import numpy as np
from tqdm import tqdm_notebook as tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from model import *

class DQN:
    
    def __init__(self, params=None):
        
        self.max_steps = 1000
        if params:
            self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay, self.learning_rate= params
        
        else:
            self.gamma = 0.95    # discount rate
            self.epsilon = 1.0  # exploration rate
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
    
    def train(self, env, num_episodes, batch_size):
        self.action_size = env.action_space.n
        
        # Initialize Replay buffer
        self.memory = deque(maxlen=2000)
        
        # Initialize Q Function
        self.q_network = Model(self.action_size)
        self.q_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        # Logging
        rewards = np.zeros(num_episodes)
        ep_lengths = np.zeros(num_episodes)
        
        pbar = tqdm(range(num_episodes), desc="Episode #: ")
        for ep in pbar:
            
            # Reset env
            state_t = env.reset()
            state_t = np.reshape(state_t, (1, state_t.size))
            done = False
            while not done:
                
                # Choose Action
                action_t = self.__act(state_t)
                
                # Step environment
                state_t_prime, reward, done, _ = env.step(action_t)
                state_t_prime = np.reshape(state_t_prime, (1, state_t_prime.size))
                
                # Store transition in replay buffer
                self.__remember(state_t, action_t, reward, state_t_prime, done)
                
                # Sample a batch from replay buffer
                if len(self.memory) > batch_size:
                    self.__replay(batch_size)
                    
                
                rewards[ep] += reward
                ep_lengths[ep] += 1
                
                if ep > 0:
                    pbar.set_description(f"Ave. Reward: {np.mean(rewards[:ep]):.2f}, Epsilon: {self.epsilon:.2f}")
                
                # Stop episode if done
                if done:
                    break
                    
        return rewards, ep_lengths
                    
    def test(self, env, num_episodes):
        # Logging
        rewards = np.zeros(num_episodes)
        ep_lengths = np.zeros(num_episodes)
        
        pbar = tqdm(range(num_episodes))
        for ep in pbar: 
            
            # Reset env
            state_t = env.reset()
            done = False
            for t in range(self.max_steps):
                
                # Choose Action
                action_t = self.__act(state_t)
                
                # Step environment
                state_t_prime, reward, done, _ = env.step(action_t)
                
                rewards[ep] += reward
                ep_lengths[ep] += 1
                
                if ep > 0:
                    pbar.set_description(f"Ave. Reward: {np.mean(rewards[:ep])}")
                
                # Stop episode if done
                if done:
                    break
                    
                
        
        
        return rewards, ep_lengths

    def __remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def __act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.q_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    def __replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.q_network.predict(next_state)[0]))
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, name):
        self.q_network.load_weights(name)

    def save(self, name):
        self.q_network.save_weights(name)