#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 05:30:56 2021

@author: adelprete
"""
import numpy as np

def policy_eval(env, gamma, pi, V, maxIters, threshold, plot=False, nprint=1000):
    ''' Policy evaluation algorithm 
        env: environment used for evaluating the policy
        gamma: discount factor
        pi: policy to evaluate
        V: initial guess of the Value table
        maxIters: max number of iterations of the algorithm
        threshold: convergence threshold
        plot: if True it plots the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # IMPLEMENT POLICY EVALUATION HERE
    
    # Iterate for at most maxIters loops
    for i in range(maxIters):
        # You can make a copy of the V table using np.copy
        V_old = np.copy(V)
        # The number of states is env.nx
        for x in range(env.nx):
            # Use env.reset(x) to set the robot state
            env.reset(x)
            # To simulate the system use env.step(u) which returns the next state and the cost
            if (pi is callable):
                u = pi(env, x)
            else:
                u = pi[x]
            x_new, cost = env.step(u)
            # Update V-Table with Bellman's equation
            V[x] = cost + gamma * V_old[x_new]
        
        # Check for convergence using the difference between the current and previous V table
        err = np.max(np.abs(V - V_old))
        if (err < threshold):
            print("Policy evaluation has converged with error: ", err)
            return V
            
        # You can plot the V table with the function env.plot_V_table(V)
        if (i % nprint == 0):
            print("Policy evaluation - Iter", i, "error", err)
            if (plot):
                env.plot_V_table(V)

    # At the env return the V table
    return V