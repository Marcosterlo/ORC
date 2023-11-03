#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:53:57 2021

@author: adelprete
"""
import numpy as np
from numpy.linalg import norm, solve

# If jacobian=True we want to compute the derivatives too, sensitivities

def rk1(x, h, u, t, ode, jacobian=False):
    if(jacobian==False):
        dx = ode.f(x, u, t)
        x_next = x + h*dx
        return x_next, dx

    (f, f_x, f_u) = ode.f(x, u, t, jacobian=True)
    dx = f
    x_next = x + h*f
    
    nx = x.shape[0]
    I = np.identity(nx)    
    phi_x = I + h*f_x
    phi_u = h * f_u
    return x_next, dx, phi_x, phi_u
'''
def rk2(x, h, u, t, ode):
    return x_next, dx

def rk2heun(x, h, u, t, ode):
    return x_next, dx

def rk3(x, h, u, t, ode):
    return x_next, dx
'''
def rk4(x, h, u, t, ode, jacobian=False):
    if(not jacobian):
        k1 = ode.f(x, u, t)
        k2 = ode.f(x + 0.5 * h * k1, u, t + 0.5 * h) 
        k3 = ode.f(x + 0.5 * h * k2, u, t + 0.5 * h) 
        k4 = ode.f(x + h * k3, u, t + h) 
        dx = 1.0/6.0*(k1 + 2*k2 + 2*k3 + k4)
        x_next = x + h * dx
        return x_next, dx

    (k1, f_x1, f_u1) = ode.f(x, u, t, jacobian=True)
    (k2, f_x2, f_u2) = ode.f(x + 0.5 * h * k1, u, t, jacobian=True)
    (k3, f_x3, f_u3) = ode.f(x + 0.5 * h * k2, u, t, jacobian=True)
    (k4, f_x4, f_u4) = ode.f(x + h * k3, u, t, jacobian=True)

    k1_x = f_x1
    k2_x = f_x2 @ (np.eye(x.shape[0]) + 0.5 * h * k1_x)
    k3_x = f_x3 @ (np.eye(x.shape[0]) + 0.5 * h * k2_x)
    k4_x = f_x4 @ (np.eye(x.shape[0]) + h * k3_x)

    phi_x = np.eye(x.shape[0]) + h * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0

    k1_u = f_u1
    k2_u = f_u2 + f_x2 @ (0.5 * h * k1_u)
    k3_u = f_u3 + f_x3 @ (0.5 * h * k2_u)
    k4_u = f_u4 + f_x4 @ (h * k3_u)

    phi_u = h * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0

    dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    x_next = x + h * dx
 
    return x_next, dx, phi_x, phi_u

'''
def semi_implicit_euler(x, h, u, t, ode):
    return x_next, dx

def implicit_euler(x, h, u, t, ode):
    return x_next, dx
'''