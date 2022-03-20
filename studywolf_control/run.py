from arms.three_link.arm import Arm
from controllers.ilqr import Control
import numpy as np
import matplotlib.pyplot as plt

dt = 1e-2
arm = Arm(dt=dt)

controller = Control(kp=100, kv=10)
controller.gen_target(arm)
controller.arm = arm

x0 = np.zeros(arm.DOF * 2)
U = np.zeros((controller.tN, arm.DOF))
X, U, cost = controller.ilqr(x0, U)

print(arm.x, controller.target)

plt.plot(X)
plt.show()
