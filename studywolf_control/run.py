import sys

sys.path.append("/home/tom/repos/control/studywolf_control/controllers")

from arms.three_link.arm import Arm
from controllers.problem import ReacherProblem
from controllers.ilqr import Control
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


problem = ReacherProblem()

controller = Control(problem)

x0 = np.zeros(problem.num_states)
U = np.random.random((controller.tN, problem.num_controls))
X, U, cost = controller.ilqr(x0, U)

print(problem.arm.x, problem.target)
plt.plot(X)
plt.show()
