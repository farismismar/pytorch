#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:09:07 2020

@author: farismismar
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Environment parameters
    # Number of slots per radio frame
    # Pilot data.    
    # Target SNR
    # Equalizer
    # Quantizer resolution
    
class radio_environment(gym.Env):
    '''    
        Observation: 
            Type: Box(2)
            Num Observation                                    Min      Max
            0   Re(h)                                          -1       1
            
        Actions:
            Type: Discrete(3)
            Num	Action
            0	Increase real
            1   Decrease real
            2	Increase imag
            3   Decrease imag
            
    '''     
    def __init__(self, N, T, SNR, random_state=None):
        self.num_actions = 4
        self.num_observations = None # Not needed; it will be computed from gym.spaces.Box()
        self.seed(random_state) # initializes the local random generator
        self.error_target = 0.5 # Error-free channel estimation
        self.varepsilon = 0.005 # step of increase/decrease of h.
        self.M = 4 # QPSK
        self.b = np.inf
        self.f_c = 3.5e9
        self.v = 2
        self.sc_spacing = 15e3
        self.symbols_per_subframe = 12
        self.equalizer = 'MMSE'
        self.true_h = []
        
        self.N = N
        self.T = T
        self.SNR = SNR
        self.mse = None
        # self.ber = None    

        # for Reinforcement Learning
        self.step_count = 0
        self.reward_min = -5
        self.reward_max = 100        
        
        bounds_lower = np.array([
            -1,
            -1])

        bounds_upper = np.array([
            1,
            1])

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here                 
        self.state = None
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
    def reset(self):
        
        # Obtain a new channel
        # Rayleigh-fading channel, which is a 
        # unity gain channel with i.i.d. Gaussian entries and zero mean.
        self.true_h = [1./np.sqrt(2) * self.np_random.normal(loc=0, scale=1),
                      1./np.sqrt(2) * self.np_random.normal(loc=0, scale=1)                      
                      ]
        
        # Initialize a channel at random.
        self.state = [1./np.sqrt(2) * self.np_random.normal(loc=0, scale=1),
                      1./np.sqrt(2) * self.np_random.normal(loc=0, scale=1)                      
                      ]
        
        self.pilot_signals, self.operational_signals = self.generate_pilot_signals(self.N, self.T, self.SNR)

        self.step_count = 0

        return np.array(self.state)
    
    def step(self, action):
        if not isinstance(action, int):
            action = int(action)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        reward = 0
        h_hat_I, h_hat_Q = state

        self.step_count += 1
        
        # Power control and beam change
        if (action == 0):
            h_hat_I += self.varepsilon
            reward += 2
        elif (action == 1):
            h_hat_I -= self.varepsilon
            reward += 1     
        elif (action == 2):
            h_hat_Q += self.varepsilon
            reward += 2
        elif (action == 3):
            h_hat_Q -= self.varepsilon
            reward += 1     
        if (action > self.num_actions - 1):
            print('WARNING: Invalid action played!')
            reward = 0
            return [], 0, False, True

        # keep track of state and BER:        
        h_hat = h_hat_I + 1j * h_hat_Q
        h_hat = h_hat / np.abs(h_hat) ** 2 # Normalize h_hat
        
        true_h = self.true_h[0] + 1j * self.true_h[1]
        
        # error = self._retrieve_ber(h_hat, self.pilot_signals)
        # self.ber = error
        error = np.mean(np.abs(true_h - h_hat) ** 2)
        self.mse = error

        # Did we find a FEASIBLE NON-DEGENERATE solution?
        done = (error >= 0) and \
                (error <= self.error_target)
                
        abort = False

        # Update the state.
        self.state = (np.real(h_hat), np.imag(h_hat))

        if abort == True:
            done = False
            reward = self.reward_min
        elif done:
            reward += self.reward_max

        return np.array(self.state), reward, done, abort

   
    def _retrieve_ber(self, h_hat, pilot_signals):
        
        # Take h_hat, equalize it, apply it on r = G_b(hx + n) + d pilot, and see if you got the right
        # pilot
        df = pd.DataFrame()
        df['r'] = pilot_signals['r_I'] + 1j * pilot_signals['r_Q']
        
        # 6) Equalization (in digital domain))
        if (self.equalizer == 'ZF'):
            w = h_hat / (np.conj(h_hat) * h_hat)
        elif (self.equalizer == 'MMSE'):
            noise_power = 10 ** (-self.SNR / 10.)
            w = h_hat / (noise_power + np.conj(h_hat) * h_hat)
        elif (self.equalizer == 'Matched'):
            w = h_hat / np.abs(h_hat)
        else:
            w = 1.
    
        x_hat_l = np.conj(w) * df['r']
        
        # Equalized quantized received signal
        df['x_hat_I'] = np.real(x_hat_l)
        df['x_hat_Q'] = np.imag(x_hat_l)
        
        print('Entering Bussgang decomposition')        
        #G_b, d, e = bussgang_decomposition(df, b)
        print('Leaving Bussgang decomposition')
    
        return 0

    
    def generate_pilot_signals(self, N, T, SNR, shuffle=False):
        noise_power = 10 ** (-SNR / 10.)
        df = self.compute_received_PSK_data(noise_power)
        Np = int(T*df.shape[0])
        
#        symbol_power = np.mean(df['x_I'] ** 2 + df['x_Q'] ** 2)
        
        X_pilot = df.groupby('m').apply(lambda x: x.loc[np.random.choice(x.index, Np//self.M, replace=False), :])
        X_pilot.index = X_pilot.index.droplevel('m')
        X_operational = df[~df.index.isin(X_pilot.index)].reset_index(drop=True)
        
        X_pilot = X_pilot.reset_index(drop=True)
        
        if shuffle:
            X_pilot = X_pilot.sample(frac=1, random_state=self.random_state)
            X_operational = X_operational.sample(frac=1, random_state=self.random_state)
            
        assert(X_pilot.shape[0] + X_operational.shape[0] == df.shape[0])

        return X_pilot, X_operational    
    
    
    def _compute_centroids_PSK(self):
        # This function computes the centroids which is the 
        # essence of the coherent detection
        # This function is correct
        
        M = self.M
        # This is the transmitted data
        centroids = pd.DataFrame(columns=['m', 'x_I', 'x_Q'])
    
        for m in np.arange(M):
            centroids = centroids.append({'m': m,
                                          'x_I': np.sqrt(1 / 2) * np.cos(2*np.pi/M*m + np.pi/M),
                                          'x_Q': np.sqrt(1 / 2) * np.sin(2*np.pi/M*m + np.pi/M)},
                                          ignore_index=True)
        
        # Normalize the transmitted symbols
        signal_power = np.mean(centroids['x_I'] ** 2 + centroids['x_Q'] ** 2)
        centroids.iloc[:, 1:] /= np.sqrt(signal_power)
        
        centroids.loc[:, 'm'] = centroids.loc[:, 'm'].astype(int)
        
        return centroids


    def compute_received_PSK_data(self, noise_power):
        c = 3e8
        centroids = self._compute_centroids_PSK()
        
        t_C = c / (self.v * self.f_c)
        S_0 = 1e-3
        sc_spacing_0 = 15e3

        T_S = t_C / S_0 * self.sc_spacing / sc_spacing_0 # number of subframes available in coherence time
            
        number_of_subframes = np.ceil(self.N / self.symbols_per_subframe).astype(int)
    
        print(f'{self.N} QPSK symbols correspond to {number_of_subframes} subframes (T_S = {T_S} subframes).')
        
        # 1) Data generator in baseband: x_l using QPSK reference symbols
        df = pd.DataFrame(np.tile(centroids.values, self.N // self.M).reshape(-1, centroids.shape[1]))
        df.columns = centroids.columns
        df['m'] = df['m'].astype(int)
        
        # 2) Channel
        h_I, h_Q = self.true_h
        h = h_I + 1j * h_Q
        
        x_l = df['x_I'] + 1j * df['x_Q']
        y_l = h * x_l
    
        # Obtain the signal prior to the additive noise introduction
        # This step spreads the channel value across the OFDM symbol
        # That is, the channel appears constant across N QPSK symbols, but changes
        # every coherence time
        df['h_I'] = np.kron(np.real(h), np.ones(self.N))
        df['h_Q'] = np.kron(np.imag(h), np.ones(self.N))
        
        df['hx_I'] = np.real(y_l)
        df['hx_Q'] = np.imag(y_l)
        
        noise = np.random.normal(0, np.sqrt(noise_power/2.), size=x_l.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2.), size=x_l.shape) 
        y_l += noise # y = hx + n
    
        # Now obtain the signal from the branches
        y_I = np.real(y_l)
        y_Q = np.imag(y_l)
        
        # The received signal before quantization
        df['y_I'] = y_I
        df['y_Q'] = y_Q
            
        # Compute noise statistics
        df.loc[:, 'n_I'] = df.loc[:, 'y_I'] - df.loc[:, 'hx_I']
        df.loc[:, 'n_Q'] = df.loc[:, 'y_Q'] - df.loc[:, 'hx_Q']
        
        # 4) Quantize data per branch onto r_l
        if (self.b < np.inf):
            #r_I_quant = bbit_adc_quantization(y_I, b=b, max_iteration=100)
            #r_Q_quant = bbit_adc_quantization(y_Q, b=b, max_iteration=100)
            #r_l = r_I_quant + 1j * r_Q_quant
            #r_l = r_l.values.flatten()
            
            # The quantized received signal
            #df['r_I'] = np.real(r_l)
            #df['r_Q'] = np.imag(r_l)
            True
        else:  
            #  full resolution, no need for quantization.
            df['r_I'] = df['y_I'] 
            df['r_Q'] = df['y_Q']
  
    
        return df