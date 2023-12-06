#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np

def mc_policy_eval(env, gamma, pi, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # create a vector N to store the number of times each state has been visited
    N = np.zeros(env.nx)
    # create a vector C to store the cumulative cost associated to each state
    C = np.zeros(env.nx)
    # create a vector V to store the Value
    V = np.zeros(env.nx)
    # create a list V_err to store history of the error between real and estimated V table
    V_err = []
    X = np.zeros(maxEpisodeLength+1)
    costs = np.zeros(maxEpisodeLength)
    
    # for each episode
    for i in range(nEpisodes):
        # reset the environment to a random state
        env.reset()
        # keep track of the states visited in this episode
        X[0] = env.x
        # keep track of the costs received at each state in this episode

        # simulate the system using the policy pi   
        for t in range (maxEpisodeLength):
            u = pi(env, X[t])
            X[t+1], costs[t] = env.step(u)
        # Update the V-Table by computing the cost-to-go J backward in time        

        J = 0
        for k in range(maxEpisodeLength-1, -1, -1):
            J = costs[k] + gamma * J
            x = int(X[k])
            N[x] += 1
            C[x] += J
            V[x] = C[x] / N[x]

        # compute V_err as: mean(abs(V-V_real))
        V_err.append(np.mean(np.abs(V -V_real)))

        if (i % nprint == 0):
            print("Iter", i, "V_err=", V_err[-1])
            if (plot):
                env.plot_V_table(V)
    
    return V, V_err


def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # make a copy of V0 using np.copy(V0)
    V = np.copy(V0)
    # create a list V__err to store the history of the error between real and estimated V table
    V_err = []
    # for each episode
    for k in range(1, nEpisodes + 1):
        # reset environment to random initial state
        env.reset()
        for t in range(maxEpisodeLength):
            # simulate the system using the policy pi
            x = env.x
            if (callable(pi)):
                u = pi(env, x)
            else:
                u = pi[x]
            x_next, cost = env.step(u)

            # at each simulation step update the Value of the current state         
            TD_target = cost + gamma * V[x_next]
            V[x] += learningRate * (TD_target - V[x]) 

    # compute V_err as: mean(abs(V-V_real))
    V_err.append(np.mean(np.abs(V-V_real)))
    if not k%nprint:
        print("Iter", k, "done")
        print("Mean|V_td - V_real|=%.5f"%V_err[-1])
        if (plot):
            env.plot_V_table(V)
    
    return V, V_err