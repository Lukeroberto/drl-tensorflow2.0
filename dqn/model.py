import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class Model(tf.keras.Model):
    # Model for DQN
    def __init__(self, output_dim, input_dim=None):
        super().__init__('mlp_policy')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(64, activation='relu')
        self.q_vals = kl.Dense(self.output_dim, activation='linear')
    
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        hidden = self.hidden1(x)
        hidden = self.hidden2(hidden)
        q_vals = self.q_vals(hidden)
        
        return q_vals