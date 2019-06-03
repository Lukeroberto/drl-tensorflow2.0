import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl

class Model(tf.keras.Model):
    # Q network 
    def __init__(self, output_dim, input_dim=None):
        super().__init__('mlp_policy')
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Two Hidden Layers
        self.hidden1 = kl.Dense(16, activation='relu')
        self.hidden2 = kl.Dense(8, activation='relu')
        
        # Outputs Q Values
        self.q_vals = kl.Dense(self.output_dim, activation='linear')
    
    def call(self, inputs):
        # Forward Pass
        hidden = self.hidden1(inputs)
        hidden = self.hidden2(hidden)
        q_vals = self.q_vals(hidden)
        
        return q_vals