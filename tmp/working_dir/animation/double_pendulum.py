import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import pandas as pd

import matplotlib.animation as animation

q1 = np.array(pd.read_csv("q1.csv").values.tolist()).flatten()
q2 = np.array(pd.read_csv("q2.csv").values.tolist()).flatten()

L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
t_stop = len(q1)*0.01  # how many seconds to simulate
history_len = len(q1)  # how many trajectory points to display

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-2.5, 2.5), ylim=(-0.1, 3))
ax.set_aspect('equal')
ax.grid()

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
trace_1l, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
trace_1u, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
trace_2l, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
trace_2u, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    thisx1 = [0, np.sin(q1[i])]
    thisy1 = [0, -np.cos(q1[i])]

    thisx2 = [np.sin(q1[i]), np.sin(q1[i]) + np.sin(-q1[i] - q2[i])]
    thisy2 = [-np.cos(q1[i]), -np.cos(q1[i]) + np.cos(-q1[i] - q2[i])]

    history_x = np.sin(q1[:i]) + np.sin(-q1[:i] - q2[:i])
    history_y = -np.cos(q1[:i]) + np.cos(-q1[:i] - q2[:i])

    lowerx1 = [0, 1.5*np.sin(3/4*np.pi)]
    lowery1 = [0, 1.5*-np.cos(3/4*np.pi)]

    upperx1 = [0, 1.5*np.sin(5/4*np.pi)]
    uppery1 = [0, 1.5*-np.cos(5/4*np.pi)]

    lowerx2 = [np.sin(q1[i]), np.sin(q1[i]) + 1.5*np.sin(q1[i] + np.pi + 3/4*np.pi)]
    lowery2 = [-np.cos(q1[i]), -np.cos(q1[i]) -1.5*np.cos(q1[i] + np.pi + 3/4*np.pi)]

    upperx2 = [np.sin(q1[i]), np.sin(q1[i]) + 1.5*np.sin(q1[i] + np.pi +  5/4*np.pi)]
    uppery2 = [-np.cos(q1[i]), -np.cos(q1[i]) -1.5*np.cos(q1[i] + np.pi + 5/4*np.pi)]

    line1.set_data(thisx1, thisy1)
    line2.set_data(thisx2, thisy2)
    trace.set_data(history_x, history_y)
    trace_1l.set_data(lowerx1, lowery1)
    trace_1u.set_data(upperx1, uppery1)
    trace_2l.set_data(lowerx2, lowery2)
    trace_2u.set_data(upperx2, uppery2)
    time_text.set_text(time_template % (i*dt))
    return line1, line2, trace, trace_1l, trace_1u, trace_2l, trace_2u, time_text


ani = animation.FuncAnimation(
    fig, animate, len(q1), interval=dt*1000, blit=True)
plt.show()

ani.save(filename="double_pendulum_term.gif", writer="pillow")
