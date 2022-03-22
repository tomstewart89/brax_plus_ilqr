import sys

sys.path.append("/home/tom/repos/control/studywolf_control/controllers")

from arms.three_link.arm import Arm
from controllers.ilqr import Control
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 1e-2
arm = Arm(dt=dt)

controller = Control(kp=100, kv=10)
controller.arm = arm
controller.target = np.random.random(size=(2,)) * np.sum(arm.L) * 0.75

x0 = np.zeros(arm.DOF * 2)
U = np.zeros((controller.tN, arm.DOF))
Xs, Us, cost = controller.ilqr(x0, U)

fig, ax = plt.subplots()
lines = ax.plot(Us[-1])


def animate(i):
    for line, X in zip(lines, Us[i % len(Xs)].T):
        line.set_ydata(X)  # update the data.
    return lines


ani = animation.FuncAnimation(fig, animate, interval=500, blit=True)

plt.show()
