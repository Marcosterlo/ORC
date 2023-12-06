#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

def value_iteration(env, gamma, V, maxIters, value_thr, plot=False, nprint=1000):
    ''' Policy iteration algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        value_thr: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # IMPLEMENT VALUE ITERATION HERE
    
    # Create an array to store the Q value of different controls
    Q = np.zeros(env.nu)
    # Iterate for at most maxIters loops
    for i in range(maxIters):
        if not i%nprint and plot:
            env.plot_V_table(V)

        # You can make a copy of current Value table using np.copy()
        V_old = np.copy(V)
        # The number of states is env.nx
        for x in range(env.nx):
            # The number of controls is env.nu
            for u in range(env.nu):
                # You can set the state of the robot using env.reset(x)
                env.reset(x)
                # You can simulate the robot using: x_next,cost = env.step(u)
                x_next, cost = env.step(u)
                Q[u] = cost + gamma * V[x_next]
                # You can get the minimum/maximum of an array using np.min() / np.max()
            V[x] = np.min(Q)
    # You can get the absolute values of the element of an array using np.abs()
        V_err = np.max(np.abs(V - V_old))
        if (V_err < value_thr):
            print("Value iteration converged after", i, "iterations with error", V_err)
            print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V))
            if (plot):
                env.plot_V_table(V)
            return V
        
        if not i%nprint:
            print("VI - Iter", i, "done")
            print("|V - V_old|=%.5f"%V_err)
    # Check convergence based on how much the Value has changed since the previous loop
    # At the end return the Value table
    print("Value iteration did NOT converge in %d iters. Error"%i, V_err)
    return V