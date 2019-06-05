import random
from collections import deque
import numpy as np
from tqdm import tqdm_notebook as tqdm
from datetime import datetime

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from model import *

class DQN:
    
    def __init__(self, params=None):
        
        self.max_steps = 50
        if params:
            self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay, self.learning_rate= params
            self.epsilon_max = params[1]
        
        else:
            self.gamma = 0.95    # discount rate
            self.epsilon = 1.0  # exploration rate
            self.epsilon_max = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
    
    def train(self, env, num_episodes, batch_size):
        self.action_size = env.action_space.n
        
        # Initialize Replay buffer
        self.memory = deque(maxlen=10000)
        
        rewards = np.zeros(num_episodes)
        ep_lengths = np.zeros(num_episodes)
        
        # Initialize Q Function
        self.q_network = Model(self.action_size)
        self.q_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        # Logging
        now = datetime.now()
        self.log_dir = "tf_logs/" + now.strftime("%Y%m%d-%H%M") + "/"
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir + "fit/", update_freq="epochs") 
        
        
        self.file_writer = tf.summary.create_file_writer(self.log_dir + "/metrics")
        self.file_writer.set_as_default()

        pbar = tqdm(range(num_episodes))
        for ep in pbar:
            
            # Reset env
            state_t = env.reset()
            state_t = np.reshape(state_t, (1, np.size(state_t)))

            done = False
            while not done:
                
                # Choose Action
                action_t = self.__act(state_t)
                
                # Step environment
                state_t_prime, reward, done, _ = env.step(action_t)
                state_t_prime = np.reshape(state_t_prime, (1, np.size(state_t_prime)))
                
                # Store transition in replay buffer
                self.__remember(state_t.astype(np.float32), action_t, reward, state_t_prime.astype(np.float32), done)
                
                # Log scalars
                rewards[ep] += reward
                ep_lengths[ep] += 1
                tf.summary.scalar('reward', data=rewards[ep], step=ep)
                tf.summary.scalar('episode length', data=ep_lengths[ep], step=ep)
                
                # Sample a batch from replay buffer
                if len(self.memory) > batch_size:
                    self.__replay(batch_size)
                    
                
                # Stop episode if done
                if done or ep_lengths[ep] == self.max_steps:
                    break
                    
            # Linear Schedule
            new_eps = self.epsilon - (self.epsilon_max - self.epsilon_min) / self.epsilon_decay
            self.epsilon = max(self.epsilon_min, new_eps)

            if ep > 0:
                pbar.set_description(f"Cum. Reward: {rewards.sum()}, Epsilon: {self.epsilon:.2f}")
#                 pbar.set_description(f"Ave. Reward: {np.mean(rewards[max(ep - 25, 0):min(ep + 25, num_episodes)]):.2f}, Epsilon: {self.epsilon:.2f}")
                    
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
#             target_f = self.q_network.predict(state)
#             target_f[0][action] = target
            estimate = self.q_network.predict(state)[0][action]
            self.q_network.fit(np.array([estimate]), 
                               np.array([target]), 
                               epochs=1, 
                               verbose=0, 
                               callbacks=[self.tensorboard_callback])
        
    def load(self, name):
        self.q_network.load_weights(name)

    def save(self, name):
        self.q_network.save_weights(name)