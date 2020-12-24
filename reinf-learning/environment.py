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
from numpy import linalg as LA
import scipy.constants

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Environment parameters
    # cell radius
    # UE movement speed
    # BS max tx power
    # Center frequency
    # Antenna heights
    # Number of ULA antenna elements on BS
    # Number of multipaths
    # DFT-based beamforming codebook
    # Probability of LOS transmission
    # Target SINR and minimum SINR
    # Oversampling factor

class radio_environment(gym.Env):
    '''    
        Observation: 
            Type: Box(4)
            Num Observation                                    Min      Max
            0   User1 server X                                 -r       r
            1   User1 server Y                                 -r       r
            2   Serving BS Power                               5        40W
            3   BF codebook index for Serving                  0        M-1
            
        Actions:
            Type: Discrete(3)
            Num	Action
            0	Power up by 1 dB using a new beam index
            1   Power down by 1 dB using a new beam index
            2	Power up by 1 dB using same beam index
            3   Power down by 1 dB using same beam index
            
    '''     
    def __init__(self, random_state=None):
        self.num_actions = 4
        self.num_observations = None # Not needed; it will be computed from gym.spaces.Box()
        self.seed(random_state) # initializes the local random generator
        self.speed = 1 # km/h.
        self.M_ULA = 4
        self.cell_radius = 150 # in meters.
        self.min_sinr = -3 # in dB
        self.sinr_target = 12 # dB
        self.max_tx_power = 40 # in Watts        
        self.f_c = 3.5e9 # Hz
        self.p_interference = 0.05 
        self.G_ant_no_beamforming = 11 # dBi
        self.prob_LOS = 0.4 # Probability of LOS transmission
        
        self.c = scipy.constants.c

        self.power_changed1 = False # did the BS power legitimally change?        
        self.bf_changed1 = False # did the BS power legitimally change?
        
        # Where are the base stations?
        self.x_bs_1, self.y_bs_1 = 0, 0
        
        # for Beamforming
        self.use_beamforming = True
        self.k_oversample = 1 # oversampling factor
        self.Np = 2 # from 3 to 5 for mmWave
        self.F = np.zeros([self.M_ULA, self.k_oversample*self.M_ULA], dtype=complex)
        self.theta_n = scipy.constants.pi * np.arange(start=0., stop=1., step=1./(self.k_oversample*self.M_ULA))
        
        # Beamforming codebook F
        for n in np.arange(self.k_oversample*self.M_ULA):
            f_n = self._compute_bf_vector(self.theta_n[n])
            self.F[:,n] = f_n
        self.f_n_bs1 = None  # The index in the codebook for serving BS

        # for Reinforcement Learning
        self.step_count = 0
        self.reward_min = -5
        self.reward_max = 10
        
        bounds_lower = np.array([
            -self.cell_radius,
            -self.cell_radius,
            1,
            0])

        bounds_upper = np.array([
            self.cell_radius,
            self.cell_radius,
            self.max_tx_power,
            self.k_oversample*self.M_ULA - 1])

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here 
                
        self.state = None
        self.received_sinr_dB = None
        self.serving_transmit_power_dB = None
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
    def reset(self):
        # Initialize f_n of cell
        self.f_n_bs1 = self.np_random.randint(self.M_ULA)
                
        self.state = [self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      np.round(self.np_random.uniform(low=1, high=self.max_tx_power), 2),
                      self.f_n_bs1
                      ]
        
        self.power_changed1 = False
        self.bf_changed1 = False 
        self.step_count = 0

        return np.array(self.state)
    
    def step(self, action):
        if not isinstance(action, int):
            action = int(action)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        state = self.state
        reward = 0
        x_ue_1, y_ue_1, pt_serving, f_n_bs1 = state

        self.step_count += 1
        
        # Power control and beam change
        if (action == 0):
            pt_serving *= 10**(1/10.)
            if self.use_beamforming:
                f_n_bs1 = (f_n_bs1 + 1) % (self.k_oversample * self.M_ULA)
                self.bf_changed1 = True
            self.power_changed1 = True
            reward += 2
        elif (action == 1):
            pt_serving *= 10**(-1/10.)
            if self.use_beamforming:
                f_n_bs1 = (f_n_bs1 + 1) % (self.k_oversample * self.M_ULA)
                self.bf_changed1 = True    
            self.power_changed1 = True
            reward += 2
        # Power control and same beam
        elif (action == 2):
            pt_serving *= 10**(0.5/10.)
            self.power_changed1 = True
            self.bf_changed1 = False
        elif (action == 3):
            pt_serving *= 10**(-0.5/10.)
            self.power_changed1 = True
            self.bf_changed1 = False
        if (action > self.num_actions - 1):
            print('WARNING: Invalid action played!')
            reward = 0
            return [], 0, False, True
        
        pt_serving = np.round(pt_serving, 2)
        
        # move the UEs at a speed of v, in a random direction
        v = self.speed * 5./18 # in m/sec
        theta_1, theta_2 = self.np_random.uniform(low=-scipy.constants.pi, high=scipy.constants.pi, size=2)
        
        dx_1 = v * math.cos(theta_1)
        dy_1 = v * math.sin(theta_1)

        # Move UE 1, but within cell radius
        x_ue_1 += dx_1
        y_ue_1 += dy_1
        
        if x_ue_1 > self.cell_radius:
            x_ue_1 = self.cell_radius
        elif x_ue_1 < -self.cell_radius:
            x_ue_1 = -self.cell_radius
            
        if y_ue_1 > self.cell_radius:
            y_ue_1 = self.cell_radius
        elif y_ue_1 < -self.cell_radius:
            y_ue_1 = -self.cell_radius
        
        # Update the beamforming codebook index
        self.f_n_bs1 = f_n_bs1
                
        received_power, received_sinr = self._compute_rf(x_ue_1, y_ue_1, pt_serving)
            
        # keep track of quantities...
        self.received_sinr_dB = received_sinr 
        self.serving_transmit_power_dBm = 10*np.log10(pt_serving*1e3)

        # Did we find a FEASIBLE NON-DEGENERATE solution?
        done = (pt_serving <= self.max_tx_power) and (pt_serving >= 0) and \
                (received_sinr >= self.min_sinr) and self.power_changed1 and self.bf_changed1 and \
                (received_sinr >= self.sinr_target)
                
        abort = (pt_serving > self.max_tx_power) or (received_sinr < self.min_sinr) or \
                (received_sinr > 35) # consider more than 35 dB SINR is too high.

        # Update the state.
        self.state = (x_ue_1, y_ue_1, pt_serving, f_n_bs1)
     
        # # Reward is how close we are to target
        # reward += int(received_sinr - self.sinr_target) * 2
        
        if abort == True:
            done = False
            reward = self.reward_min
        elif done:
            reward += self.reward_max

        return np.array(self.state), reward, done, abort

    def _compute_bf_vector(self, theta):
        c = scipy.constants.c
        
        wavelength = c / self.f_c
        
        d = wavelength / 2. # antenna spacing 
        k = 2. * scipy.constants.pi / wavelength
    
        exponent = 1j * k * d * math.cos(theta) * np.arange(self.M_ULA)
        
        f = 1. / math.sqrt(self.M_ULA) * np.exp(exponent)
        
        # Test the norm square... is it equal to unity? YES.
    #    norm_f_sq = LA.norm(f, ord=2) ** 2
     #   print(norm_f_sq)
    
        return f

    def _compute_channel(self, x_ue, y_ue, x_bs, y_bs):
        # Np is the number of paths p
        PLE_L = 2
        PLE_N = 4
        G_ant = 3 # dBi for beamforming mmWave antennas
        
        # Override the antenna gain if no beamforming
        if self.use_beamforming == False:
            G_ant = self.G_ant_no_beamforming
            
        # theta is the steering angle.  Sampled iid from unif(0,pi).
        theta = self.np_random.uniform(low=0, high=scipy.constants.pi, size=self.Np)
    
        is_mmWave = (self.f_c > 25e9)
        
        if is_mmWave:
            path_loss_LOS = 10 ** (self._path_loss_mmWave(x_ue, y_ue, PLE_L, x_bs, y_bs) / 10.)
            path_loss_NLOS = 10 ** (self._path_loss_mmWave(x_ue, y_ue, PLE_N, x_bs, y_bs) / 10.)
        else:
            path_loss_LOS = 10 ** (self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)
            path_loss_NLOS = 10 ** (self._path_loss_sub6(x_ue, y_ue, x_bs, y_bs) / 10.)
            
        # Bernoulli for p
        alpha = np.zeros(self.Np, dtype=complex)
        p = self.np_random.binomial(1, self.prob_LOS)
        
        if (p == 1):
            self.Np = 1
            alpha[0] = 1. / math.sqrt(path_loss_LOS)
        else:
            ## just changed alpha to be complex in the case of NLOS
            alpha = (self.np_random.normal(size=self.Np) + 1j * self.np_random.normal(size=self.Np)) / math.sqrt(path_loss_NLOS)
                
        rho = 1. * 10 ** (G_ant / 10.)
        
        # initialize the channel as a complex variable.
        h = np.zeros(self.M_ULA, dtype=complex)
        
        for p in np.arange(self.Np):
            a_theta = self._compute_bf_vector(theta[p])
            h += alpha[p] / rho * a_theta.T # scalar multiplication into a vector
        
        h *= math.sqrt(self.M_ULA)
        return h

    def _compute_rf(self, x_ue, y_ue, pt_bs1):
        T = 290 # Kelvins
        B = 15000 # Hz
        k_Boltzmann = 1.38e-23
        
        noise_power = k_Boltzmann*T*B # this is in Watts

        # Without loss of generality, the base station is at the origin
        # The interfering base station is x = cell_radius, y = 0
        x_bs_1, y_bs_1 = self.x_bs_1, self.y_bs_1

        # Now the channel h, which is a vector in beamforming.
        # This computes the channel for user in serving BS from the serving BS.
        h_1 = self._compute_channel(x_ue, y_ue, x_bs=x_bs_1, y_bs=y_bs_1) 
          
        # if this is not beamforming, there is no precoder:
        if (self.use_beamforming):
            received_power = pt_bs1 * abs(np.dot(h_1.conj(), self.F[:, self.f_n_bs1])) ** 2
        else: # the gain is ||h||^2
            received_power = pt_bs1 * LA.norm(h_1, ord=2) ** 2
                
        # TODO: interference power can be a sporadic signal with a bernoulli dist.
        interference_power = self.np_random.binomial(1, self.p_interference) * 1 # in Watts
        interference_plus_noise_power = interference_power + noise_power
        received_sinr = 10*np.log10(received_power / interference_plus_noise_power)

        return [np.round(received_power, 2), np.round(received_sinr, 2)]
    
    # https://ieeexplore-ieee-org.ezproxy.lib.utexas.edu/stamp/stamp.jsp?tp=&arnumber=7522613
    def _path_loss_mmWave(self, x, y, PLE, x_bs=0, y_bs=0):
        # These are the parameters for f = 28000 MHz.
        wavelength = self.c / self.f_c
        A = 0.0671
        Nr = self.M_ULA
        sigma_sf = 9.1
        #PLE = 3.812
        
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2) # in meters
        
        fspl = 10 * np.log10(((4*scipy.constants.pi*d) / wavelength) ** 2)
        pl = fspl + 10 * np.log10(d ** PLE) * (1 - A*np.log2(Nr))
    
        chi_sigma = self.np_random.normal(0, sigma_sf) # log-normal shadowing 
        L = pl + chi_sigma
    
        return L # in dB    
        
    def _path_loss_sub6(self, x, y, x_bs=0, y_bs=0):
        f_c = self.f_c
        
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2)
        h_B = 20
        h_R = 1.5

#        print('Distance from cell site is: {} km'.format(d/1000.))
        # FSPL
        L_fspl = -10*np.log10((4.*scipy.constants.pi*self.c/f_c / d) ** 2)
        
        # COST231        
        C = 3
        a = (1.1 * np.log10(f_c/1e6) - 0.7)*h_R - (1.56*np.log10(f_c/1e6) - 0.8)
        L_cost231  = 46.3 + 33.9 * np.log10(f_c/1e6) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d/1000.) + C
    
        L = L_cost231
        
        return L # in dB