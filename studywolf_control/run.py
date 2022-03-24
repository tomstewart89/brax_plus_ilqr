import sys

sys.path.append("/home/tom/repos/control/studywolf_control/controllers")

from arms.three_link.arm import Arm
from controllers.ilqr import Control
import numpy as jnp
import matplotlib.pyplot as plt

dt = 1e-2

arm = Arm(dt=dt)

controller = Control(kp=100, kv=10)
controller.arm = arm
controller.target = jnp.random.random(size=(2,)) * jnp.sum(arm.L) * 0.75

x0 = jnp.zeros(arm.DOF * 2)
U = jnp.zeros((controller.tN, arm.DOF))
X, U, cost = controller.ilqr(x0, U)

print(arm.x, controller.target)
plt.plot(X)
plt.show()
