import brax
from jax import grad, jacrev, jit
import jax
import jax.numpy as jnp
import numpy as np
from google.protobuf import text_format

import sys

sys.path.append("/home/tom/repos/control/studywolf_control/controllers")

from arms.three_link.arm import Arm


def finite_differences(f, x, eps=1e-4):
    """Differentiate a function `f` using finite differences"""
    p = np.zeros_like(x)

    for i in range(x.size):
        p[i] = eps
        df_dxi = (f(x + p) - f(x - p)) / (2 * eps)
        p[i] = 0.0

        if i == 0:
            D = np.zeros((df_dxi.size, x.size))

        D[:, i] = df_dxi

    return D


class ReacherProblem:
    def __init__(self):
        self.sys = brax.System(text_format.Parse(ReacherProblem.config, brax.Config()))
        self.arm = Arm(dt=1e-2)

        self.sys.default_qp()

        self.num_controls = self.arm.DOF
        self.num_states = self.arm.DOF * 2
        self.num_timesteps = 50

        self.target = np.random.random(2) * np.sum(self.arm.L) * 0.75

        self.lt_x = jit(grad(self.running_cost, argnums=0))
        self.lt_u = jit(grad(self.running_cost, argnums=1))
        self.lt_xx = jit(jacrev(grad(self.running_cost, argnums=0), argnums=0))
        self.lt_uu = jit(jacrev(grad(self.running_cost, argnums=1), argnums=1))
        self.lt_ux = jit(jacrev(grad(self.running_cost, argnums=1), argnums=0))
        self.lT_x = jit(grad(self.terminal_cost))
        self.lT_xx = jit(jacrev(grad(self.terminal_cost)))

    def step(self, x, u):
        brax.QP()

    def step(self, x, u):
        # set the arm position to x
        self.arm.reset(q=x[: self.arm.DOF], dq=x[self.arm.DOF :])

        # apply the control signal
        self.arm.apply_torque(u)

        # get the system state from the arm
        return np.hstack([np.copy(self.arm.q), np.copy(self.arm.dq)])

    def running_cost(self, x, u):
        return 1e-2 * (u**2).sum()

    def terminal_cost(self, x):
        """the final state cost function"""

        l1 = self.arm.L[0] * jnp.array([jnp.cos(jnp.sum(x[:1])), jnp.sin(jnp.sum(x[:1]))])
        l2 = self.arm.L[1] * jnp.array([jnp.cos(jnp.sum(x[:2])), jnp.sin(jnp.sum(x[:2]))])
        l3 = self.arm.L[2] * jnp.array([jnp.cos(jnp.sum(x[:3])), jnp.sin(jnp.sum(x[:3]))])

        return 1e5 * jnp.sum((l1 + l2 + l3 - self.target) ** 2) + 1e5 * jnp.sum(x[3:] ** 2)

    def f_x(self, x, u):
        return finite_differences(lambda d: self.step(d, u), x)

    def f_u(self, x, u):
        return finite_differences(lambda d: self.step(x, d), u)

    config = """
    bodies {
        name: "ground"
        colliders {
        plane {
        }
        }
        mass: 1.0
        inertia {
        x: 1.0
        y: 1.0
        z: 1.0
        }
        frozen {
        all: true
        }
    }
    bodies {
        name: "body0"
        colliders {
        position {
            x: 0.05
        }
        rotation {
            y: 90.0
        }
        capsule {
            radius: 0.01
            length: 0.12
        }
        }
        inertia {
        x: 1.0
        y: 1.0
        z: 1.0
        }
        mass: 0.035604715
    }
    bodies {
        name: "body1"
        colliders {
        position {
            x: 0.05
        }
        rotation {
            y: 90.0
        }
        capsule {
            radius: 0.01
            length: 0.12
        }
        }
        colliders {
        position { x: .11 }
        sphere {
            radius: 0.01
        }
        }
        inertia {
        x: 1.0
        y: 1.0
        z: 1.0
        }
        mass: 0.035604715
    }
    bodies {
        name: "target"
        colliders {
        position {
        }
        sphere {
            radius: 0.009
        }
        }
        inertia {
        x: 1.0
        y: 1.0
        z: 1.0
        }
        mass: 1.0
        frozen { all: true }
    }
    joints {
        name: "joint0"
        parent: "ground"
        child: "body0"
        parent_offset {
        z: 0.01
        }
        child_offset {
        }
        rotation {
        y: -90.0
        }
        angle_limit {
            min: -360
            max: 360
        }
    }
    joints {
        name: "joint1"
        parent: "body0"
        child: "body1"
        parent_offset {
        x: 0.1
        }
        child_offset {
        }
        rotation {
        y: -90.0
        }
        angle_limit {
        min: -360
        max: 360
        }
    }
    actuators {
        name: "joint0"
        joint: "joint0"
        strength: 25.0
        torque {
        }
    }
    actuators {
        name: "joint1"
        joint: "joint1"
        strength: 25.0
        torque {
        }
    }
    collide_include {
    }
    gravity {
        z: -9.81
    }
    dt: 0.02
    substeps: 4
    frozen {
        position {
        z: 1.0
        }
        rotation {
        x: 1.0
        y: 1.0
        }
    }
    dynamics_mode: "pbd"
    """
