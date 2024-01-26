import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

T = 0.1                  # OCP horizion
dt = 0.01               # OCP time step
max_iter = 100          # Maximum iteration per point

terminal_constraint_on = 1
initial_state = np.array([np.pi*6/7, -0.5])
q_target = 5/4 * np.pi

mpc_step = 500

lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
lowerVelocityLimit = -10
upperVelocityLimit = 10
lowerControlBound = -9.81
upperControlBound = 9.81

w_q = 1e2
w_v = 1e-1
w_u = 1e-4

scaler = StandardScaler()

dataframe = pd.read_csv("../datasets/single_data_14641.csv")
labels = dataframe['viable']
dataset = dataframe.drop('viable', axis=1)
train_size = 0.8
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=17)
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

def init_scaler():
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    return scaler_mean, scaler_std

# Function to create states array in a grid
def grid_states(n_pos, n_vel):
    n_ics = n_pos * n_vel
    possible_q = np.linspace(lowerPositionLimit, upperPositionLimit, num=n_pos)
    possible_v = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=n_vel)
    state_array = np.zeros((n_ics, 2))

    j = k = 0
    for i in range (n_ics):
        state_array[i,:] = np.array([possible_q[j], possible_v[k]])
        k += 1
        if (k == n_vel):
            k = 0
            j += 1

    return n_ics, state_array

# Function to create states array taken from a uniform distribution
def random_states(n_states):
    state_array = np.zeros((n_states, 2))

    for i in range(n_states):
        possible_q = (upperPositionLimit - lowerPositionLimit) * np.random.random_sample() + lowerPositionLimit
        possible_v = (upperVelocityLimit - lowerVelocityLimit) * np.random.random_sample() + lowerVelocityLimit
        state_array[i,:] = np.array([possible_q, possible_v])
    
    return n_states, state_array
