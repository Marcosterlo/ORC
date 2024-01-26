import numpy as np
import casadi 
import working_dir.dynamics.double_pendulum_dynamics as double_pendulum_dynamics
import multiprocessing
import ocp_double_pendulum_conf as conf
import pandas as pd
import matplotlib.pyplot as plt
import time

class OcpDoublePendulum:

    def __init__(self):
        self.T = conf.T             # OCP horizon
        self.dt = conf.dt           # time step
        # self.w_q1 = conf.w_q1       # Weights       
        self.w_v1 = conf.w_v1       
        self.w_u1 = conf.w_u1       
        # self.w_q2 = conf.w_q2       
        self.w_v2 = conf.w_v2       
        self.w_u2 = conf.w_u2       

    def solve(self, x_init, X_guess = None, U_guess = None):
        self.N = int(self.T/self.dt)
        self.opti = casadi.Opti()

        self.q1 = self.opti.variable(self.N+1)
        self.q2 = self.opti.variable(self.N+1)
        self.v1 = self.opti.variable(self.N+1)
        self.v2 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        self.u2 = self.opti.variable(self.N)
        q1 = self.q1
        q2 = self.q2
        v1 = self.v1
        v2 = self.v2
        u1 = self.u1
        u2 = self.u2

        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0, i])
                self.opti.set_initial(q2[i], X_guess[1, i])
                self.opti.set_initial(v1[i], X_guess[2, i])
                self.opti.set_initial(v2[i], X_guess[3, i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(q2[i], x_init[2])
                self.opti.set_initial(v1[i], x_init[1])
                self.opti.set_initial(v2[i], x_init[3])
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0, i])
                self.opti.set_initial(u2[i], U_guess[1, i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            if (i<self.N):                           # Check necessary since at the last step it doesn't make sense to consider the input
                self.running_costs[i] += self.w_u1 * u1[i]**2 + self.w_u2 * u2[i]**2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)

        # Dynamics constraint
        for i in range(self.N):
            # Next state computation with dynamics
            x_next = double_pendulum_dynamics.f(np.array([q1[i], q2[i], v1[i], v2[i]]), np.array([u1[i], u2[i]]))
            # Dynamics imposition
            self.opti.subject_to(q1[i+1] == x_next[0])
            self.opti.subject_to(q2[i+1] == x_next[1])
            self.opti.subject_to(v1[i+1] == x_next[2])
            self.opti.subject_to(v2[i+1] == x_next[3])
        
        # Initial state constraint
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(v2[0] == x_init[3])

        # Bounds constraints
        for i in range(self.N+1):
            # Position bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit1, q1[i], conf.upperPositionLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit2, q2[i], conf.upperPositionLimit2))

            # Velocity bounds
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit1, v1[i], conf.upperVelocityLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit2, v2[i], conf.upperVelocityLimit2))
            
            if (i<self.N):
                # Control bounds
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound1, u1[i], conf.upperControlBound1))
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound2, u2[i], conf.upperControlBound2))
        
        return self.opti.solve()

def check(state_array):
    # n_ics, state_array = conf.grid_states(conf.npos1, conf.nvel1, conf.npos2, conf.nvel2)
    n_ics = len(state_array)
    state_array = np.array(state_array)

    ocp = OcpDoublePendulum()

    start = time.time()
    
    viable = []
    no_viable = []
    
    for i in range(n_ics):
        x = state_array[i, :]
        try:
            sol = ocp.solve(x)
            viable.append([x[0], x[2], x[1], x[3]])
            print("Feasible initial state found:", x)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("Non feasible initial state found:", x)
                no_viable.append([x[0], x[2], x[1], x[3]])
            else:
                print("Runtime error:", e)

    viable_states = np.array(viable)
    no_viable_states = np.array(no_viable)

    # Stop keeping track of time
    end = time.time()

    # Time in nice format
    tot_time = end-start
    seconds = int(tot_time % 60)
    minutes = int((tot_time - seconds) / 60)       
    hours = int((tot_time - seconds - minutes*60) / 3600)
    print("Total elapsed time:", hours, "h", minutes, "min", seconds, "s")

    # Unify both viable and non viable states with a flag to show wether they're viable or not
    viable_states = np.column_stack((viable_states, np.ones(len(viable_states), dtype=int)))
    no_viable_states = np.column_stack((no_viable_states, np.zeros(len(no_viable_states), dtype=int)))
    if len(viable_states) == 0:
        dataset = no_viable_states
    elif len(no_viable_states) == 0:
        dataset = viable_states
    else:
        dataset = np.concatenate((viable_states, no_viable_states))

    # Create a DataFrame starting from the final array
    columns = ['q1', 'q2', 'v1', 'v2', 'viable']
    df = pd.DataFrame(dataset, columns=columns)

    return df
