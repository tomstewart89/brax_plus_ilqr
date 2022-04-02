import sys

sys.path.append("/home/tom/repos/control/studywolf_control/controllers")

from arms.three_link.arm import Arm
from controllers.ilqr import Control
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

arm = Arm(dt=1e-2)
target = np.random.random(size=(2,)) * np.sum(arm.L) * 0.75


@jax.jit
def running_cost(x, u):
    return 1e-2 * (u**2).sum()


@jax.jit
def terminal_cost(x):
    """the final state cost function"""

    l1 = arm.L[0] * jnp.array([jnp.cos(jnp.sum(x[:1])), jnp.sin(jnp.sum(x[:1]))])
    l2 = arm.L[1] * jnp.array([jnp.cos(jnp.sum(x[:2])), jnp.sin(jnp.sum(x[:2]))])
    l3 = arm.L[2] * jnp.array([jnp.cos(jnp.sum(x[:3])), jnp.sin(jnp.sum(x[:3]))])

    return 1e5 * jnp.sum((l1 + l2 + l3 - target) ** 2) + 1e5 * jnp.sum(x[3:] ** 2)


def plant_dynamics(x, u):
    """simulate a single time step of the plant, from
    initial state x and applying control signal u

    x np.array: the state of the system
    u np.array: the control signal
    """
    # set the arm position to x
    arm.reset(q=x[: arm.DOF], dq=x[arm.DOF :])

    # apply the control signal
    arm.apply_torque(u)

    # get the system state from the arm
    return np.hstack([np.copy(arm.q), np.copy(arm.dq)])


controller = Control(running_cost, terminal_cost, plant_dynamics, arm.DOF * 2, arm.DOF)

x0 = np.zeros(arm.DOF * 2)
U = np.random.random((controller.tN, arm.DOF))
X, U, cost = controller.ilqr(x0, U)

print(arm.x, target)
plt.plot(X)
plt.show()


# import brax
# import jax
# import jax.numpy as jp
# from google.protobuf import text_format

# with open("/home/tom/Desktop/reacher.proto", "r") as f:
#     config_proto = f.read()

# config = text_format.Parse(config_proto, brax.Config())
# sys = brax.System(config)

# qpos = sys.default_angle()
# qvel = jp.zeros(sys.num_joint_dof)
# qp = sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
# qp = brax.QP(jp.array(qp.pos), jp.array(qp.rot), jp.array(qp.vel), jp.array(qp.ang))


# @jax.jit
# def wrap(act):
#     new_state, _ = sys.step(qp, act)
#     return jp.mean(new_state.pos)


# a = jp.zeros(3)
# f_u = jax.grad(wrap)
# print(f_u(a))
# print(f_u(a))
# print(f_u(a))
# print(f_u(a))
