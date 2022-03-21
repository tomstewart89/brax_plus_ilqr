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
        self.wp = 1e5  # terminal position cost weight
        self.wv = 1e5  # terminal velocity cost weight

    def cost(self, x, u):
        """the intermediate state cost function"""
        # compute cost
        dof = u.shape[0]
        num_states = x.shape[0]

        l = np.sum(u**2)

        # compute derivatives of cost
        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))

        l_u = 2 * u
        l_uu = 2 * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        # returned in an array for easy multiplication by time step
        return l, l_x, l_xx, l_u, l_uu, l_ux

    def cost_final(self, x):
        """the final state cost function"""

        num_states = x.shape[0]
        dof = self.arm.DOF

        l = self.wp * np.sum((self.arm.x - self.target) ** 2) + self.wv * np.sum(x[dof:] ** 2)

        l_x = np.zeros(num_states)
        l_x[:dof] = self.wp * self.endpoint_error_grad(x[:dof])
        l_x[dof:] = 2 * self.wv * x[dof:]

        l_xx = np.zeros((num_states, num_states))
        l_xx[:dof, :dof] = self.wp * finite_differences(self.endpoint_error_grad, x[:dof])
        l_xx[dof:, dof:] = 2 * self.wv * np.eye(dof)

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
        xnext = np.hstack([np.copy(self.arm.q), np.copy(self.arm.dq)])

        # calculate the change in state
        return ((xnext - x) / self.arm.dt).squeeze()

    def ilqr(self, x0, U):
        """use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function

        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        tN = U.shape[0]  # number of time steps
        dof = self.arm.DOF  # number of degrees of freedom of plant
        num_states = dof * 2  # number of states (position and velocity)
        dt = self.arm.dt  # time step

        lamb = 1.0  # regularization parameter
        sim_new_trajectory = True

        for ii in range(self.max_iter):

            if sim_new_trajectory == True:
                # simulate forward using the current control trajectory
                X, cost = self.simulate(x0, U)
                oldcost = np.copy(cost)  # copy for exit condition check

                # now we linearly approximate the dynamics, and quadratically
                # approximate the cost function so we can use LQR methods

                # for storing linearized dynamics
                # x(t+1) = f(x(t), u(t))
                f_x = np.zeros((tN, num_states, num_states))  # df / dx
                f_u = np.zeros((tN, num_states, dof))  # df / du
                # for storing quadratized cost function
                l = np.zeros((tN, 1))  # immediate state cost
                l_x = np.zeros((tN, num_states))  # dl / dx
                l_xx = np.zeros((tN, num_states, num_states))  # d^2 l / dx^2
                l_u = np.zeros((tN, dof))  # dl / du
                l_uu = np.zeros((tN, dof, dof))  # d^2 l / du^2
                l_ux = np.zeros((tN, dof, num_states))  # d^2 l / du / dx
                # for everything except final state
                for t in range(tN - 1):
                    # x(t+1) = f(x(t), u(t)) = x(t) + dx(t) * dt
                    # linearized dx(t) = np.dot(A(t), x(t)) + np.dot(B(t), u(t))
                    # f_x = np.eye + A(t)
                    # f_u = B(t)
                    A, B = self.plant_dynamics_grad(X[t], U[t])
                    f_x[t] = np.eye(num_states) + A * dt
                    f_u[t] = B * dt

                    (l[t], l_x[t], l_xx[t], l_u[t], l_uu[t], l_ux[t]) = self.cost(X[t], U[t])
                    l[t] *= dt
                    l_x[t] *= dt
                    l_xx[t] *= dt
                    l_u[t] *= dt
                    l_uu[t] *= dt
                    l_ux[t] *= dt
                # aaaand for final state
                l[-1], l_x[-1], l_xx[-1] = self.cost_final(X[-1])

                sim_new_trajectory = False

            # optimize things!
            # initialize Vs with final state cost and set up k, K
            V = l[-1].copy()  # value function
            V_x = l_x[-1].copy()  # dV / dx
            V_xx = l_xx[-1].copy()  # d^2 V / dx^2
            k = np.zeros((tN, dof))  # feedforward modification
            K = np.zeros((tN, dof, num_states))  # feedback gain

            # NOTE: they use V' to denote the value at the next timestep,
            # they have this redundant in their notation making it a
            # function of f(x + dx, u + du) and using the ', but it makes for
            # convenient shorthand when you drop function dependencies

            # work backwards to solve for V, Q, k, and K
            for t in range(tN - 2, -1, -1):

                # NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

                # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
                Q_x = l_x[t] + np.dot(f_x[t].T, V_x)
                # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
                Q_u = l_u[t] + np.dot(f_u[t].T, V_x)

                # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
                # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.

                # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
                Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t]))
                # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
                Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
                # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
                Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

                # Calculate Q_uu^-1 with regularization term set by
                # Levenberg-Marquardt heuristic (at end of this loop)
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                Q_uu_inv = np.dot(Q_uu_evecs, np.dot(np.diag(1.0 / Q_uu_evals), Q_uu_evecs.T))

                # 5b) k = -np.dot(Q_uu^-1, Q_u)
                k[t] = -np.dot(Q_uu_inv, Q_u)
                # 5b) K = -np.dot(Q_uu^-1, Q_ux)
                K[t] = -np.dot(Q_uu_inv, Q_ux)

                # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
                # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
                V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
                # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
                V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

            Unew = np.zeros((tN, dof))
            # calculate the optimal change to the control trajectory
            xnew = x0.copy()  # 7a)
            for t in range(tN - 1):
                # use feedforward (k) and feedback (K) gain matrices
                # calculated from our value function approximation
                # to take a stab at the optimal control signal
                Unew[t] = U[t] + k[t] + np.dot(K[t], xnew - X[t])  # 7b)
                # given this u, find our next state
                xnew += self.plant_dynamics(xnew, Unew[t]) * dt  # 7c)

            # evaluate the new trajectory
            Xnew, costnew = self.simulate(x0, Unew)

            # Levenberg-Marquardt heuristic
            if costnew < cost:
                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X = np.copy(Xnew)  # update trajectory
                U = np.copy(Unew)  # update control signal
                oldcost = np.copy(cost)
                cost = np.copy(costnew)

                sim_new_trajectory = True  # do another rollout

                # print("iteration = %d; Cost = %.4f;"%(ii, costnew) +
                #         " logLambda = %.1f"%np.log(lamb))
                # check to see if update is small enough to exit
                if ii > 0 and ((abs(oldcost - cost) / cost) < self.eps_converge):
                    print(
                        "Converged at iteration = %d; Cost = %.4f;" % (ii, costnew)
                        + " logLambda = %.1f" % np.log(lamb)
                    )
                    break

            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor
                # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
                if lamb > self.lamb_max:
                    print(
                        "lambda > max_lambda at iteration = %d;" % ii
                        + " Cost = %.4f; logLambda = %.1f" % (cost, np.log(lamb))
                    )
                    break

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
            X[t + 1] = self.plant_dynamics(X[t], U[t]) * dt + X[t]
            l, _, _, _, _, _ = self.cost(X[t], U[t])
            cost = cost + dt * l

        # Adjust for final cost, subsample trajectory
        l_f, _, _ = self.cost_final(X[-1])
        cost = cost + l_f

        return X, cost
