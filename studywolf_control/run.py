from controllers.problem import ReacherProblem
from controllers.ilqr import Control
import brax.jumpy as jp
import matplotlib.pyplot as plt


problem = ReacherProblem()
controller = Control(problem)

x0 = jp.zeros(problem.num_states)
U = jp.zeros((problem.num_timesteps, problem.num_controls))
X, U, cost = controller.ilqr(x0, U)

print(problem.arm.x - problem.target)
plt.plot(X)
plt.show()
