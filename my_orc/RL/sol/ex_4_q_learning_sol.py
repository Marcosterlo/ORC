#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def q_learning(env, gamma, Q, nEpisodes, maxEpisodeLength, 
               learningRate, exploration_prob, exploration_decreasing_decay,
               min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' Q learning algorithm:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    # Make a copy of the initial Q table guess
    Q_old = np.copy(Q)
    epsilon = exploration_prob
    # for every episode
    for k in range(nEpisodes):
        # reset the state to a random value
        x = env.reset()
        # Cost of currend episode
        J = 0
        # simulate the system for maxEpisodeLength steps
        for t in range(maxEpisodeLength):
            # with probability exploration_prob take a random control input
            # epsilon-greedy implementation
            if (uniform(0,1) < epsilon):
                u = randint(0, env.nu) # random integer between 0 and the number of possible actions
            else:
                # otherwise take a greedy control
                u = np.argmin(Q[x,:])
            # Simulate the system
            x_next, l = env.step(u)
            # Compute reference Q-value at state x
            Q_target = l + gamma * np.min(Q[x_next, :])
            # Update Q-Table with the given learningRate
            Q[x, u] += learningRate * (Q_target - Q[x, u])
            x = x_next
            # keep track of the cost to go
            # J += cost if no discount factor
            J += (gamma**t) * l

        # update the exploration probability with an exponential decay: eps = exp(-decay*episode)
        epsilon = np.exp(-exploration_decreasing_decay * k)
        epsilon = max(epsilon, min_exploration_prob) # We saturate to the lower bound

        # Update cost to go history
        h_ctg.append(J)
        if (k % nprint == 0):
            print("Episode", k, "[Q - Q_old]", np.max(np.abs(Q-Q_old)), "eps", epsilon)
            Q_old = np.copy(Q)
            if (plot):
                # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
                V, pi = compute_V_pi_from_Q(env, Q)
                env.plot_V_table(V)
                env.plot_policy(pi)
    
    return Q, h_ctg