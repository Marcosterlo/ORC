import numpy as np
import casadi
import working_dir.dynamics.double_pendulum_dynamics as double_pendulum_dynamics
import working_dir.double_pendulum.mpc_double_pendulum_conf as conf
from working_dir.nn.double_neural_network import create_model

class MpcDoublePendulum:

    def __init__(self):
        self.T = conf.T                             # MPC horizon
        self.dt = conf.dt                           # time step
        self.w_q1 = conf.w_q1                       # 1 Position weight
        self.w_u1 = conf.w_u1                       # 1 Input weight
        self.w_v1 = conf.w_v1                       # 1 Velocity weight
        self.w_q2 = conf.w_q2                       # 2 Position weight
        self.w_u2 = conf.w_u2                       # 2 Input weight
        self.w_v2 = conf.w_v2                       # 2 Velocity weight
        self.N = int(self.T/self.dt)                # I initalize the Opti helper from casadi
        self.model = create_model(4)                # Template of NN
        self.model.load_weights("../nn/active_learning/iterata2.h5")
        self.weights = self.model.get_weights()
        self.mean, self.std = conf.init_scaler()
    
    def nn_to_casadi(self, params, x):
        out = np.array(x)
        iteration = 0

        for param in params:
            param = np.array(param.tolist())

            if iteration % 2 == 0:
                out = out @ param
            else:
                out = param + out
                for i, item in enumerate(out):
                    out[i] = casadi.fmax(0., casadi.MX(out[i]))

            iteration += 1

        return casadi.MX(out[0])
        
    def solve(self, x_init, X_guess = None, U_guess = None):
        self.opti = casadi.Opti()                   # N is the size of the vector we want to realize, the number of steps we want to compute. We create a vector containing all the states of size N+1, 
                                                    # We create a vector of control inputs of size N, one less than the list of states since final control input doesn't have any importance
        # Casadi variables declaration
        self.q1 = self.opti.variable(self.N+1)       
        self.v1 = self.opti.variable(self.N+1)
        self.u1 = self.opti.variable(self.N)
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1
        self.q2 = self.opti.variable(self.N+1)       
        self.v2 = self.opti.variable(self.N+1)
        self.u2 = self.opti.variable(self.N)
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2
        
        # State vector initialization
        if (X_guess is not None):
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], X_guess[0,i])
                self.opti.set_initial(q2[i], X_guess[1,i])
                self.opti.set_initial(v1[i], X_guess[2,i])
                self.opti.set_initial(v2[i], X_guess[3,i])
        else:
            for i in range(self.N+1):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(q2[i], x_init[1])
                self.opti.set_initial(v1[i], x_init[2])
                self.opti.set_initial(v2[i], x_init[3])
        
        # Control input vector initalization
        if (U_guess is not None):
            for i in range(self.N):
                self.opti.set_initial(u1[i], U_guess[0,i])
                self.opti.set_initial(u2[i], U_guess[1,i])

        # Choosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.max_iter)}
        self.opti.solver("ipopt", opts, s_opst)

        # Target position
        q1_target = conf.q1_target
        q2_target = conf.q2_target

        # Cost definition
        self.cost = 0
        self.running_costs = [None,]*(self.N+1)      # Defining vector of Nones that will contain running cost values for each step
        for i in range(self.N+1):
            self.running_costs[i] = self.w_v1 * v1[i]**2 + self.w_v2 * v2[i]**2
            self.running_costs[i] += (self.w_q1 * (q1_target - q1[i])**2 + self.w_q2 * (q2_target - q2[i])**2)
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
        self.opti.subject_to(q2[0] == x_init[1])
        self.opti.subject_to(v1[0] == x_init[2])
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
        
        # Terminal constraint (NN)
        state = [(q1[self.N] - self.mean[0])/self.std[0], (q2[self.N] - self.mean[1])/self.std[1], (v1[self.N] - self.mean[2])/self.std[2], (v2[self.N] - self.mean[3])/self.std[3]]
        if conf.terminal_constraint_on:
            self.opti.subject_to(self.nn_to_casadi(self.weights, state) > 1)

        return self.opti.solve()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import pandas as pd
    
    mpc = MpcDoublePendulum()

    initial_state = conf.initial_state
    actual_trajectory = []
    actual_inputs = []

    actual_trajectory.append(initial_state)

    mpc_step = conf.mpc_step
    new_state_guess = np.zeros((4, mpc.N+1))
    new_input_guess = np.zeros((2, mpc.N))

    # First run
    sol = mpc.solve(initial_state)
    actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v1[1]), sol.value(mpc.v2[1])]))
    actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
    for i in range(mpc.N):
        new_state_guess[0, i] = sol.value(mpc.q1[i+1])
        new_state_guess[1, i] = sol.value(mpc.q2[i+1])
        new_state_guess[2, i] = sol.value(mpc.v1[i+1])
        new_state_guess[3, i] = sol.value(mpc.v2[i+1])
    for i in range(mpc.N-1):
        new_input_guess[0, i] = sol.value(mpc.u1[i+1])
        new_input_guess[1, i] = sol.value(mpc.u2[i+1])
    new_state_guess[:,mpc.N] = np.zeros(4)
    new_input_guess[:,mpc.N-1] = np.zeros(2)
    
    for i in range(mpc_step):
        try:
            sol = mpc.solve(actual_trajectory[i+1], new_state_guess, new_input_guess)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("")
                print("======================================")
                print("MPC stopped due to infeasible problem")
                print("======================================")
                print("")
                # print(mpc.opti.debug.show_infeasibilities())
                break
            else:
                print(e)
                break
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.q2[1]), sol.value(mpc.v1[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))
        for k in range(mpc.N):
            new_state_guess[0, k] = sol.value(mpc.q1[k+1])
            new_state_guess[1, k] = sol.value(mpc.q2[k+1])
            new_state_guess[2, k] = sol.value(mpc.v1[k+1])
            new_state_guess[3, k] = sol.value(mpc.v2[k+1])
        for j in range(mpc.N-1):
            new_input_guess[0, j] = sol.value(mpc.u1[j+1])
            new_input_guess[1, j] = sol.value(mpc.u2[j+1])
        new_state_guess[:,mpc.N] = np.zeros(4)
        new_input_guess[:,mpc.N-1] = np.zeros(2)
        print("Step", i+1, "out of", mpc_step, "done")

    ## PLOTTA
    q1 = []
    q2 = []
    v1 = []
    v2 = []

    for i, state in enumerate(actual_trajectory):
        q1.append(actual_trajectory[i][0])
        q2.append(actual_trajectory[i][1])

    for i, state in enumerate(actual_trajectory):
        v1.append(actual_trajectory[i][2])
        v2.append(actual_trajectory[i][3])

    pd.DataFrame(q1).to_csv("../animation/q1.csv", index=False)
    pd.DataFrame(q2).to_csv("../animation/q2.csv", index=False)

    fig, ax = plt.subplots()

    def init(): 
        return []
    
    for i in range(2):
        ax.clear()
        n_grid = 101 
        v1_act = v1[i]
        v2_act = v2[i]
        possible_q1 = np.linspace(conf.lowerPositionLimit1, conf.upperPositionLimit1, num=n_grid)
        possible_q2 = np.linspace(conf.lowerPositionLimit2, conf.upperPositionLimit2, num=n_grid)
        state_array = np.zeros((n_grid**2,4))
        to_test = np.zeros((n_grid**2,4))

        j = k = 0
        for s in range(n_grid**2):
            state_array[s,:] = np.array([possible_q1[k], possible_q2[j], v1_act, v2_act])
            to_test[s,:] = np.array([(possible_q1[k] - mpc.mean[0])/mpc.std[0], (possible_q2[j] - mpc.mean[1])/mpc.std[1], (v1_act - mpc.mean[2])/mpc.std[2], (v2_act - mpc.mean[3])/mpc.std[3]])
            j+=1
            if (j==n_grid):
                j=0
                k+=1
        
        # to_test = conf.scaler.transform(state_array)
        label_pred = mpc.model.predict(to_test)
        viable_states = []
        no_viable_states = []

        for k, label in enumerate(label_pred):
            if label:
                viable_states.append(state_array[k,:])
            else:
                no_viable_states.append(state_array[k,:])
        
        viable_states = np.array(viable_states)
        no_viable_states = np.array(no_viable_states)

        if len(viable_states) != 0:
            ax.scatter(viable_states[:,0], viable_states[:,1], c='r')
        if len(no_viable_states) != 0:
            ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b')
        ax.scatter(q1[0:i], q2[0:i], c='g')
        ax.set_xlabel('q1 [rad]')
        ax.set_ylabel('q2 [rad]')

    def update(i):
        ax.clear()
        n_grid = 101 
        v1_act = v1[i]
        v2_act = v2[i]
        possible_q1 = np.linspace(conf.lowerPositionLimit1, conf.upperPositionLimit1, num=n_grid)
        possible_q2 = np.linspace(conf.lowerPositionLimit2, conf.upperPositionLimit2, num=n_grid)
        state_array = np.zeros((n_grid**2,4))
        to_test = np.zeros((n_grid**2,4))

        j = k = 0
        for s in range(n_grid**2):
            state_array[s,:] = np.array([possible_q1[k], possible_q2[j], v1_act, v2_act])
            to_test[s,:] = np.array([(possible_q1[k] - mpc.mean[0])/mpc.std[0], (possible_q2[j] - mpc.mean[1])/mpc.std[1], (v1_act - mpc.mean[2])/mpc.std[2], (v2_act - mpc.mean[3])/mpc.std[3]])
            j+=1
            if (j==n_grid):
                j=0
                k+=1
        
        # to_test = conf.scaler.transform(state_array)
        label_pred = mpc.model.predict(to_test)
        viable_states = []
        no_viable_states = []

        for k, label in enumerate(label_pred):
            if label>0.5:
                viable_states.append(state_array[k,:])
            else:
                no_viable_states.append(state_array[k,:])
        
        viable_states = np.array(viable_states)
        no_viable_states = np.array(no_viable_states)

        if len(viable_states) != 0:
            ax.scatter(viable_states[:,0], viable_states[:,1], c='r')
        if len(no_viable_states) != 0:
            ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b')
        ax.scatter(q1[0:i], q2[0:i], c='g')
        ax.set_xlabel('q1 [rad]')
        ax.set_ylabel('q2 [rad]')
        return []

    if conf.plot:
        num_frames = len(q1)
        animation = FuncAnimation(fig, update, frames=num_frames, blit=True)
        animation.save(filename="test.gif", writer="pillow")

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(q1, q2, c='r')
    ax.set_xlabel('q1 [rad]')
    ax.set_ylabel('q2 [rad]')
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(actual_inputs)
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(q1)
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(q2)
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(v1)
    plt.show()

    fig = plt.figure(figsize=(12,8))
    plt.plot(v2)
    plt.show()