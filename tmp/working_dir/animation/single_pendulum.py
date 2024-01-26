import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import sin, cos

import matplotlib.animation as animation

data = np.array(pd.read_csv("q.csv").values.tolist()).flatten()

L1 = 1.0  # length of pendulum 1 in m
t_stop = len(data)*0.01  # how many seconds to simulate
history_len = len(data)  # how many trajectory points to display

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.01
t = np.arange(0, t_stop, dt)

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-1, 1), ylim=(-0.1, 1.2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
trace_bound1, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
trace_bound2, = ax.plot([], [], '.--', lw=1, ms=2, color="green")
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    thisx = [0, np.sin(data[i])]
    thisy = [0, -np.cos(data[i])]

    lowerx = [0, 1.5*np.sin(3/4*np.pi)]
    lowery = [0, 1.5*-np.cos(3/4*np.pi)]

    upperx = [0, 1.5*np.sin(5/4*np.pi)]
    uppery = [0, 1.5*-np.cos(5/4*np.pi)]

    history_x = np.sin(data[:i])
    history_y = -np.cos(data[:i])

    line.set_data(thisx, thisy)
    trace_bound1.set_data(lowerx, lowery)
    trace_bound2.set_data(upperx, uppery)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text

ani = animation.FuncAnimation(
    fig, animate, len(data), interval=dt*1000, blit=True)
plt.show()

ani.save(filename="animation_terminal.gif", writer="pillow")
