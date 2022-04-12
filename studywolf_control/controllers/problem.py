import brax
import brax.jumpy as jp
from brax.math import rotate
from jax import grad, jacrev, jit
from google.protobuf import text_format


class ReacherProblem:
    def __init__(self):
        self.sys = brax.System(text_format.Parse(ReacherProblem.config, brax.Config()))

        self.num_controls = self.sys.num_actuators
        self.num_states = self.sys.num_joints * 2
        self.num_timesteps = 50

        self.target = jp.array([0.0, 0.1, 0.0])

        self.f_x = jit(jacrev(self.step, argnums=0))
        self.f_u = jit(jacrev(self.step, argnums=1))
        self.lt_x = jit(grad(self.running_cost, argnums=0))
        self.lt_u = jit(grad(self.running_cost, argnums=1))
        self.lt_xx = jit(jacrev(grad(self.running_cost, argnums=0), argnums=0))
        self.lt_uu = jit(jacrev(grad(self.running_cost, argnums=1), argnums=1))
        self.lt_ux = jit(jacrev(grad(self.running_cost, argnums=1), argnums=0))
        self.lT_x = jit(grad(self.terminal_cost))
        self.lT_xx = jit(jacrev(grad(self.terminal_cost)))

    def to_reduced_coordinates(self, qp):
        rel_rot = jp.vmap(brax.math.relative_quat)(qp.rot[0:2], qp.rot[1:3])
        joint_angles = jp.vmap(brax.math.quat_to_axis_angle)(rel_rot)[1]

        rel_ang = qp.ang[0:2] - qp.ang[1:3]
        joint_velocities = rel_ang[:, 2]

        return jp.concatenate([joint_angles, joint_velocities])

    def to_maximal_coordinates(self, x):
        return self.sys.default_qp(joint_angle=x[:2], joint_velocity=x[2:])

    def step(self, x, u):
        return self.to_reduced_coordinates(self.sys.step(self.to_maximal_coordinates(x), u)[0])

    def running_cost(self, x, u):
        return 1e-2 * (u**2).sum()

    def terminal_cost(self, x):
        """the final state cost function"""

        qp = self.to_maximal_coordinates(x)

        jp.array([self.sys.config.bodies[2].colliders[0].capsule.length, 0, 0])

        return 1e5 * jp.sum((self.to_maximal_coordinates(x).pos[2] - self.target) ** 2) + 1e5 * jp.sum(
            x[2:] ** 2
        )

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
