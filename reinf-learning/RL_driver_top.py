#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:13:06 2020

@author: farismismar
"""

import os
import time
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU and CPU fallback
#from QLearningAgent import QLearningAgent as QLearner

MAX_EPISODES = 100
radio_frame = 10

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # My NVIDIA GTX 1080 Ti FE GPU

output = pd.DataFrame()

os.chdir('/Users/farismismar/Desktop/deep')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlepad'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath}\usepackage{amssymb}'


def run_agent_q(env, radio_frame, plotting=True):
    global output
   
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
        print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | 0 | - dB | - W | 0 | {action}')
              
        total_reward = 0
        done = False
       
        episode_loss = []
        episode_q = []

        output_z = pd.DataFrame()
        
        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (x_t, y_t, tx_power_t, bf_index_t) = next_observation
                        
            received_sinr = env.received_sinr_dB

            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)
                           
            # Learn policy through replay.
            q, loss = agent.replay()
                      
            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward            
                            
            successful = done and (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | {timestep_index} | {received_sinr} dB | {tx_power_t} W | {total_reward} | {action} | ', end='')
    
            # Store the action, reward, and observation elements, done|aborted
            # for further postprocessing and plotting
            
            output_t = pd.Series([episode_index, timestep_index, x_t, y_t, tx_power_t, bf_index_t, action, done, abort])
            output_z = output_z.append(output_t, ignore_index=True)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()            
            
            # Update for the next time step
            action = agent.act(observation, total_reward)

        # Episode ends
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)
        output_z.loc[:, 'Reward'] = total_reward
        output_z.loc[:, 'Loss'] = loss_z
        output_z.loc[:, 'Q'] = q_z
        
        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.  Loss = {}.'.format(total_reward, loss_z))
            print(Style.RESET_ALL)
            episode_successful.append(episode_index)
            
            # Keep an eye on the best episode
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)
        
        losses.append(loss_z)
        Q_values.append(q_z)
        output = pd.concat([output, output_z], axis=0)
        
    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes or reduce target.".format(max_episodes_to_run))
    else:
        print(f'Episode {max_episode}/{MAX_EPISODES} generated the highest reward {max_reward}.')

    output.columns = ['Episode', 'Time', 'UE x', 'UE y', 'BS TX Power', 'Beam Index', 'Action', 'Done', 'Abort', 'Reward', 'Loss', 'Q']
    output.to_csv('output.csv', index=False)
    
    if plotting:
        summary = pd.DataFrame(data={'Episode': 1 + np.arange(max_episodes_to_run),
                                     'Avg. Loss': losses,
                                     'Avg. Q': Q_values})
        plot_summary(summary)


def plot_summary(df):
    fig = plt.figure(figsize=(8,5))
    
    plot1, = plt.plot(df['Episode'], df['Avg. Q'], c='blue')
    plt.grid(which='both', linestyle='--')
    
    ax = fig.gca()    
    ax_sec = ax.twinx()
    plot2, = ax_sec.plot(df['Episode'], df['Avg. Loss'], lw=2, c='red')       
    plt.xlabel('Episode')    
    ax.set_ylabel(r'$Q$')
    ax_sec.set_ylabel(r'$L$')    
    plt.legend([plot1, plot2], [r'Average $Q$', r'Average loss'],
               bbox_to_anchor=(0.1, -0.03, 0.86, 1), bbox_transform=fig.transFigure, 
               loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    

seeds = np.arange(1).tolist()

for seed in seeds:
 
    env = radio_environment(random_state=seed)
    agent = QLearner(random_state=seed)
    start_time = time.time()
    run_agent_q(env, radio_frame)
    end_time = time.time()
    
    print('Takes {:.2f} seconds.'.format(end_time - start_time))
    
########################################################################################
