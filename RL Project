# rl_project/main.py
import gym
from stable_baselines3 import DQN

# Create environment
env = gym.make('CartPole-v1')

# Create DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Train model
model.learn(total_timesteps=10000)

# Save model
model.save('dqn_cartpole')

# Load model and evaluate
model = DQN.load('dqn_cartpole')
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
