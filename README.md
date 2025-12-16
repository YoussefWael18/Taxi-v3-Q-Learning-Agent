# Taxi-v3 Reinforcement Learning

A Q-Learning implementation for solving the classic Taxi-v3 environment from OpenAI Gymnasium. This project demonstrates how an agent learns to pick up and drop off passengers efficiently using reinforcement learning.

## Overview

This project implements a **Q-Learning algorithm** to train an agent in the Taxi-v3 environment. The taxi learns to:
- Navigate a 5x5 grid
- Pick up passengers from one of four locations
- Drop them off at their destination
- Maximize rewards while minimizing penalties

## Features

- **Q-Learning Implementation**: Uses a Q-table to store state-action values
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation
- **Visual Testing**: Watch the trained agent perform in real-time
- **Configurable Hyperparameters**: Easy to adjust learning parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL-taxi.git
cd RL-taxi
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training and testing script:
```bash
python Taxi_RL.py
```

The script will:
1. Train the agent for 10,000 episodes
2. Display 5 test episodes with visual rendering
3. Show the agent's performance

##  Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.9 | Learning rate |
| `gamma` | 0.97 | Discount factor |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.9995 | Decay rate for epsilon |
| `min_epsilon` | 0.01 | Minimum exploration rate |
| `num_episodes` | 10,000 | Training episodes |
| `max_steps` | 100 | Maximum steps per episode |

## How It Works

### Q-Learning Algorithm

The agent uses the Q-learning update rule:

```
Q(s,a) ← (1-α)Q(s,a) + α[r + γ max Q(s',a')]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `α` = learning rate
- `γ` = discount factor

### State Space

The environment has **500 discrete states** representing:
- 25 taxi positions (5x5 grid)
- 5 passenger locations (4 locations + in taxi)
- 4 destination locations

### Action Space

The agent can take **6 actions**:
- 0: Move south
- 1: Move north
- 2: Move east
- 3: Move west
- 4: Pick up passenger
- 5: Drop off passenger

##  Results

After training, the agent successfully learns to:
- Navigate to the passenger's location
- Pick up the passenger
- Navigate to the destination
- Drop off the passenger

The epsilon-greedy strategy ensures the agent explores initially and gradually exploits learned knowledge.

## 🔧 Customization

You can modify the hyperparameters at the top of `Taxi_RL.py`:

```python
alpha = 0.9          
gamma = 0.97        
epsilon = 1         
epsilon_decay = 0.9995
min_epsilon = 0.01
num_episodes = 10000
max_steps = 100
```

## Dependencies

- Python 3.7+
- gymnasium
- numpy
- pygame

See `requirements.txt` for specific versions.


