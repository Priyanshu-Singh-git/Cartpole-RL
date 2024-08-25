# CartPole Reinforcement Learning with DQN

## Overview

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 environment using OpenAI Gym and TensorFlow. The agent learns to balance a pole on a cart by applying forces to the cart, aiming to prevent the pole from falling over.

## Environment

The CartPole environment is a classic problem in reinforcement learning where the objective is to keep a pole balanced on a moving cart. The environment is considered solved when the agent achieves an average reward of 195.0 over 100 consecutive episodes.

### Environment Details:

- **Observation Space**: 
  - Type: Box (4)
  - Components:
    1. Cart Position
    2. Cart Velocity
    3. Pole Angle
    4. Pole Velocity at Tip
- **Action Space**: 
  - Type: Discrete (2)
  - Actions: 
    - 0: Push cart to the left
    - 1: Push cart to the right
- **Reward**: +1 for every time step the pole remains upright.
- **Termination**: 
  - Pole Angle exceeds ±12 degrees.
  - Cart Position exceeds ±2.4 units.
  - Episode length exceeds 200 time steps.

## Implementation

The project uses TensorFlow and the `rl` module to build and train the DQN agent.

### Key Components:

- **Model Architecture**: 
  - A Sequential model with two hidden layers of 24 neurons each, using ReLU activation.
  - The output layer uses a linear activation function to predict Q-values for each action.

- **DQN Agent**: 
  - **Memory**: Uses `SequentialMemory` with a limit of 50,000 experiences and a window length of 1.
  - **Policy**: Implements the BoltzmannQPolicy for action selection.
  - **Training**: The agent is trained for 100,000 steps, with a warmup period of 10 steps.

### Code Example:

```python
import random
import numpy as np
import gym
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="human")

# Define the model
states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))

# Define the agent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)
agent.compile(Adam(learning_rate=0.001), metrics=["mae"])

# Train the agent
agent.fit(env, nb_steps=100000, visualize=True, verbose=1)

# Test the agent
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

# Close the environment
env.close()
