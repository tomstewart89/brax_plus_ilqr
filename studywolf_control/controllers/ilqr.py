"""
Copyright (C) 2016 Travis DeWolf 

Implemented from 'Control-limited differential dynamic programming'
by Yuval Tassa, Nicolas Mansard, and Emo Todorov (2014).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


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


def to_cart(r, theta):
    """Convert a polar coordinate (r,theta) into a cartesian coordinate"""
    return r * np.array([np.cos(theta), np.sin(theta)])


class Control:
    """
    A controller that implements iterative Linear Quadratic Gaussian control.
    Controls the (x,y) position of a robotic arm end-effector.
    """

    def __init__(self, n=50, max_iter=100, **kwargs):
        """
        n int: length of the control sequence
        max_iter int: limit on number of optimization iterations
        """

        self.tN = n  # number of timesteps
        self.max_iter = max_iter
        self.lamb_factor = 10
        self.lamb_max = 1000
        self.eps_converge = 0.001  # exit if relative improvement below threshold
        self.running_w = 1e-2  # running cost weight
        self.terminal_wp = 1e5  # terminal position cost weight
        self.terminal_wv = 1e5  # terminal velocity cost weight

    def running_cost(self, x, u):
        """the intermediate state cost function"""
        dof = u.shape[0]
        num_states = x.shape[0]

        l = self.running_w * np.sum(u**2)

        # compute derivatives of cost
        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))

        l_u = 2 * self.running_w * u
        l_uu = 2 * self.running_w * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        # returned in an array for easy multiplication by time step
        return l, l_x, l_xx, l_u, l_uu, l_ux

    def terminal_cost(self, x):
        """the final state cost function"""

        num_states = x.shape[0]
        dof = self.arm.DOF

        l = self.terminal_wp * np.sum((self.arm.x - self.target) ** 2) + self.terminal_wv * np.sum(
            x[dof:] ** 2
        )

        l_x = np.zeros(num_states)
        l_x[:dof] = self.terminal_wp * self.endpoint_error_grad(x[:dof])
        l_x[dof:] = 2 * self.terminal_wv * x[dof:]

        l_xx = np.zeros((num_states, num_states))
        l_xx[:dof, :dof] = self.terminal_wp * finite_differences(self.endpoint_error_grad, x[:dof])
        l_xx[dof:, dof:] = 2 * self.terminal_wv * np.eye(dof)

        # Final cost only requires these three values
        return l, l_x, l_xx

    def endpoint_error_grad(self, x):
        """Compute derivative of endpoint error"""

        end_pos = np.sum([to_cart(self.arm.L[i], np.sum(x[: i + 1])) for i in range(self.arm.DOF)], axis=0)
        error = end_pos - self.target

        error_dot = np.zeros(self.arm.DOF + 1)

        for i in reversed(range(self.arm.DOF)):
            theta = np.sum(x[: i + 1])
            error_dot[i] = 2 * self.arm.L[i] * error.dot([-np.sin(theta), np.cos(theta)]) + error_dot[i + 1]

        return error_dot[:-1]

    def plant_dynamics_grad(self, x, u):
        """calculate derivative of plant dynamics using finite differences

        x np.array: the state of the system
        u np.array: the control signal
        """

        f_x = finite_differences(lambda d: self.plant_dynamics(d, u), x)
        f_u = finite_differences(lambda d: self.plant_dynamics(x, d), u)

        return f_x, f_u

    def plant_dynamics(self, x, u):
        """simulate a single time step of the plant, from
        initial state x and applying control signal u

        x np.array: the state of the system
        u np.array: the control signal
        """
        dof = self.arm.DOF

        # set the arm position to x
        self.arm.reset(q=x[:dof], dq=x[dof:])

        # apply the control signal
        self.arm.apply_torque(u)

        # get the system state from the arm
        return np.hstack([np.copy(self.arm.q), np.copy(self.arm.dq)])

    def ilqr(self, x0, U):
        """use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function

        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        tN = U.shape[0]  # number of time steps
        dof = self.arm.DOF  # number of degrees of freedom of plant
        num_states = x0.size  # number of states (position and velocity)

        lamb = 1.0  # regularization parameter

        # for storing linearized dynamics: x(t+1) = f(x(t), u(t))
        f_x = np.zeros((tN, num_states, num_states))  # df / dx
        f_u = np.zeros((tN, num_states, dof))  # df / du

        # for storing quadratized cost function
        l = np.zeros(tN)  # immediate state cost
        l_x = np.zeros((tN, num_states))  # dl / dx
        l_xx = np.zeros((tN, num_states, num_states))  # d^2 l / dx^2
        l_u = np.zeros((tN, dof))  # dl / du
        l_uu = np.zeros((tN, dof, dof))  # d^2 l / du^2
        l_ux = np.zeros((tN, dof, num_states))  # d^2 l / du / dx

        k = np.zeros((tN, dof))  # feedforward modification
        K = np.zeros((tN, dof, num_states))  # feedback gain

        # simulate forward using the current control trajectory
        X, cost = self.simulate(x0, U)

        for i in range(self.max_iter):

            # now we linearly approximate the dynamics, and quadratically
            # approximate the cost function so we can use LQR methods

            # for everything except final state
            for t in range(tN - 1):
                f_x[t], f_u[t] = self.plant_dynamics_grad(X[t], U[t])
                l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t] = self.running_cost(X[t], U[t])

            # aaaand for final state
            l[-1], l_x[-1], l_xx[-1] = self.terminal_cost(X[-1])

            # optimize things!
            # initialize Vs with final state cost and set up k, K
            V_x = l_x[-1].copy()  # dV / dx
            V_xx = l_xx[-1].copy()  # d^2 V / dx^2

            # work backwards to solve for V, Q, k, and K
            for t in reversed(range(tN - 1)):

                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x
                Q_x = l_x[t] + f_x[t].T @ V_x  # 4a)
                Q_u = l_u[t] + f_u[t].T @ V_x  # 4b)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                Q_xx = l_xx[t] + f_x[t].T @ V_xx @ f_x[t]  # 4c)
                Q_ux = l_ux[t] + f_u[t].T @ V_xx @ f_x[t]  # 4d)
                Q_uu = l_uu[t] + f_u[t].T @ V_xx @ f_u[t]  # 4e)

                # Calculate Q_uu^-1 with regularization term set by
                # Levenberg-Marquardt heuristic (at end of this loop)
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                Q_uu_inv = Q_uu_evecs @ np.diag(1.0 / Q_uu_evals) @ Q_uu_evecs.T

                k[t] = -Q_uu_inv @ Q_u  # 5b)
                K[t] = -Q_uu_inv @ Q_ux  # 5b)

                V_x = Q_x - K[t].T @ Q_uu @ k[t]  # 6b)
                V_xx = Q_xx - K[t].T @ Q_uu @ K[t]  # 6c)

            # calculate the optimal change to the control trajectory
            Xnew = np.zeros((tN, num_states))
            Unew = np.zeros((tN, dof))
            Xnew[0] = x0
            costnew = 0

            for t in range(tN - 1):
                # use feedforward (k) and feedback (K) gain matrices
                # calculated from our value function approximation
                # to take a stab at the optimal control signal
                Unew[t] = U[t] + k[t] + K[t] @ (Xnew[t] - X[t])  # 7b)

                costnew += self.running_cost(Xnew[t], Unew[t])[0]

                # given this u, find our next state
                Xnew[t + 1] = self.plant_dynamics(Xnew[t], Unew[t])  # 7c)

            costnew += self.terminal_cost(Xnew[t])[0]

            # Levenberg-Marquardt heuristic
            if costnew < cost:
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X = Xnew  # update trajectory
                U = Unew  # update control signal

                improvement = abs(cost - costnew) / costnew
                cost = costnew

                # check to see if update is small enough to exit
                if i > 0 and improvement < self.eps_converge:
                    break

            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor

                if lamb > self.lamb_max:
                    break

        print("Finished at iteration = %d; Cost = %.4f; logLambda = %.1f" % (i, costnew, np.log(lamb)))

        return X, U, cost

    def simulate(self, x0, U):
        """do a rollout of the system, starting at x0 and
        applying the control sequence U

        x0 np.array: the initial state of the system
        U np.array: the control sequence to apply
        """
        tN = U.shape[0]
        num_states = x0.shape[0]
        dt = self.arm.dt

        X = np.zeros((tN, num_states))
        X[0] = x0
        cost = 0

        # Run simulation with substeps
        for t in range(tN - 1):
            X[t + 1] = self.plant_dynamics(X[t], U[t])
            cost += self.running_cost(X[t], U[t])[0]

        # Adjust for final cost, subsample trajectory
        cost += self.terminal_cost(X[-1])[0]

        return X, cost
