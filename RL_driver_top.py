#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:13:06 2020

@author: farismismar
"""

import os
import numpy as np
from colorama import Fore, Style

from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU and CPU fallback
#from QLearningAgent import QLearningAgent as QLearner

MAX_EPISODES = 10000

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";   # My NVIDIA GTX 1080 Ti FE GPU


os.chdir('/Users/farismismar/Desktop/deep')

def run_agent_q(env, radio_frame, plotting=False):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False
    episode_successful = [] # a list to save the good episodes
    Q_values = []   
    losses = []
    
    agent.prepare_agent(env)
    
    max_episode = -1
    max_reward = -np.inf
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
    
        print('Ep.        | eta | TS | Recv. SINR (srv) | Serv. Tx Pwr | Reward | Action')
        print('--'*54)
            
        (_, _, pt_serving, _) = observation

        action = agent.begin_episode(observation)           
        # Let us know how we did.
        print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | 0 | - dB | {pt_serving} W | 0 | {action}')
              
        total_reward = 0
        done = False
        actions = [action]
        
        sinr_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        
        episode_loss = []
        episode_q = []

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (_, _, pt_serving, _) = next_observation
                        
            received_sinr = env.received_sinr_dB
            
          #  next_observation = np.reshape(next_observation, [1, agent._state_size])
            
            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)
                           
            # Learn control policy
            loss, q = agent.replay()
                      
            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward            
                            
            successful = done and (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | {timestep_index} | {received_sinr} dB | {pt_serving} W | {total_reward} | {action} | ', end='')
    
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()            
            
            # Update for the next time step
            action = agent.act(observation, total_reward)
            
        # at the level of the episode end
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)
        
        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.  Loss = {}.'.format(total_reward, loss_z))
            print(Style.RESET_ALL)
            
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)

        
        losses.append(loss_z)
        Q_values.append(q_z)
        
        if (successful):
            episode_successful.append(episode_index)

            optimal = 'Episode {}/{} generated the highest reward {}.'.format(max_episode, MAX_EPISODES, max_reward)
            print(optimal)


    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

radio_frame = 15
seeds = np.arange(50).tolist()

for seed in seeds:
 
    env = radio_environment(random_state=seed)
    agent = QLearner(random_state=seed)

    run_agent_q(env, radio_frame)

########################################################################################
