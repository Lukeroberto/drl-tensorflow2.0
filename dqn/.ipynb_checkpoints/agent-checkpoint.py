class DQN:
    
    def __init__(self):
        pass
    
    def train(self):
        # Initialize Replay buffer
        # Initialize Q Function
        
        for ep in range(self.num_episodes):
            state_t = env.reset()
            
            for t in range(max_steps):
                # Choose Action
                action_t = self._epsilonGreedy(state_t)
                
                # Step environment
                state_t_prime, reward, done = env.step(action_t)
                
                # Store transition in replay buffer
                transition = Transition(state_t, action_t, state_t_prime, reward, done)
                self.replay_buffer.store(transition)
                
                # Sample a batch from replay buffer
                batch_x, batch_y = self.replay_buffer.sample()
                
                # Learn from experience
                self._learn(batch_x, batch_y)
    
    def test(self):
        pass
    
    def _epsilonGreedy(self, state):
        pass
    
    def _learn(X, Y):
        pass