import random
import pygame
import gymnasium as gy
import numpy as np 
import time

env=gy.make('Taxi-v3')

alpha=0.9  # learning rate 
gamma=0.97 # discount factor  
epsilon=1   # exploration (all actions will be random)
epsilon_decay=0.9995 # we will multiply the epsilion with this value to decay it by time
min_epsilon=0.01 # least epsilion we can reach 
num_episodes=10000 # how many times the agent will work
max_steps=100 # the episodes terminates if taxi went into circles

# Q-table 
# we have 500 states (25*5*4) --> 25 postitions for the taxi, 5 different coloured squares for customer, 4 diff loctations for the hotel
Q_table= np.zeros((env.observation_space.n,env.action_space.n))


def choose_actions(state):
    """
        This function takes an ction based on the state
    """
    if random.uniform(0 ,1)< epsilon: # if the random number is less than epsilon
        return env.action_space.sample()   # We take a random action 
    else:
        return np.argmax(Q_table[state,:]) # else we choose the best action from the Q-table
    

for episode in range(num_episodes):

    state,_=env.reset() # We reset the enviroment at each episode 
    done=False          # starting the enviroment

    for step in range (max_steps):
        action = choose_actions(state)  # We take the action from the choose action function
        next_state,reward,done,truncated,info=env.step(action) # Taking the next step

        old_value=Q_table[state,action] # Q-table value for the action we took in a specifc state
        next_max=np.max(Q_table[next_state,:]) # the max Q-table action

        Q_table[state,action]=(1-alpha)*old_value + alpha * (reward+gamma * next_max)
        state=next_state

        if done or truncated:
            break
        
    epsilon=max(min_epsilon,epsilon * epsilon_decay)

# Testing Section
# We change the environment to human mode so we can see the taxi moving
env=gy.make('Taxi-v3',render_mode='human')

for episode in range(5): # We run 5 episodes to watch our agent play
    state,_=env.reset() # We reset the environment for the new episode
    done=False

    print('Episode',episode)

    for step in range(max_steps): # We loop through the steps of the episode
        env.render() # We update the screen to show the current state
        
        # We choose the best action from the Q-table (Exploitation)
        action = np.argmax(Q_table[state, :])
        
        # We take the step and get the 5 return values
        next_state, reward, done, truncated, info = env.step(action)
        
        # We update our current state to the new state
        state = next_state

        # We check if the episode is finished (delivered or crashed)
        if done or truncated:
            env.render() # Show the final move
            print('Episode finished:', episode, 'With reward:', reward) 
            break
            
env.close() # We close the window and clean up