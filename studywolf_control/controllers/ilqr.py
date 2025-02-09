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
from brax.jumpy import vmap


class Control:
    """
    A controller that implements iterative Linear Quadratic Gaussian control.
    Controls the (x,y) position of a robotic arm end-effector.
    """

    def __init__(self, problem, max_iter=100):
        """
        max_iter int: limit on number of optimization iterations
        """

        self.problem = problem
        self.max_iter = max_iter
        self.lamb_factor = 5
        self.lamb_max = 10000
        self.eps_converge = 0.001  # exit if relative improvement below threshold

    def simulate(self, x0, U):
        """do a rollout of the system, starting at x0 and
        applying the control sequence U

        x0 np.array: the initial state of the system
        U np.array: the control sequence to apply
        """
        X = np.zeros((self.problem.num_timesteps, self.problem.num_states))
        X[0] = x0
        cost = 0

        # Run simulation with substeps
        for t in range(self.problem.num_timesteps - 1):
            X[t + 1] = self.problem.step(X[t], U[t])
            cost += self.problem.running_cost(X[t], U[t])

        # Adjust for final cost, subsample trajectory
        cost += self.problem.terminal_cost(X[-1])

        return X, cost

    def forward_pass(self, X, U):
        """linearly approximate the dynamics, and quadratically approximate the cost function so we can use LQR methods"""

        l_x = np.vstack([vmap(self.problem.lt_x)(X[:-1], U[:-1]), self.problem.lT_x(X[-1])[None, :]])
        l_u = np.vstack([vmap(self.problem.lt_u)(X[:-1], U[:-1]), np.zeros((1, self.problem.num_controls))])
        l_xx = np.vstack([vmap(self.problem.lt_xx)(X[:-1], U[:-1]), self.problem.lT_xx(X[-1])[None, :, :]])

        l_uu = np.vstack(
            [
                vmap(self.problem.lt_uu)(X[:-1], U[:-1]),
                np.zeros((1, self.problem.num_controls, self.problem.num_controls)),
            ]
        )

        l_ux = np.vstack(
            [
                vmap(self.problem.lt_ux)(X[:-1], U[:-1]),
                np.zeros((1, self.problem.num_controls, self.problem.num_states)),
            ]
        )

        f_x = vmap(self.problem.f_x)(X, U)
        f_u = vmap(self.problem.f_u)(X, U)

        return f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux

    def backward_pass(self, lamb, f_x, f_u, l_x, l_xx, l_u, l_uu, l_ux):

        k = np.zeros((self.problem.num_timesteps, self.problem.num_controls))  # feedforward modification
        K = np.zeros(
            (self.problem.num_timesteps, self.problem.num_controls, self.problem.num_states)
        )  # feedback gain

        # optimize things!
        # initialize Vs with final state cost and set up k, K
        V_x = l_x[-1].copy()  # dV / dx
        V_xx = l_xx[-1].copy()  # d^2 V / dx^2

        # work backwards to solve for V, Q, k, and K
        for t in reversed(range(self.problem.num_timesteps - 1)):

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

        return k, K

    def calculate_step(self, X, U, k, K):
        """calculate the optimal change to the control trajectory"""

        Xnew = np.zeros_like(X)
        Unew = np.zeros_like(U)
        Xnew[0] = X[0]
        costnew = 0

        for t in range(self.problem.num_timesteps - 1):
            # use feedforward (k) and feedback (K) gain matrices
            # calculated from our value function approximation
            # to take a stab at the optimal control signal
            Unew[t] = U[t] + k[t] + K[t] @ (Xnew[t] - X[t])  # 7b)

            costnew += self.problem.running_cost(Xnew[t], Unew[t])

            # given this u, find our next state
            Xnew[t + 1] = self.problem.step(Xnew[t], Unew[t])  # 7c)

        costnew += self.problem.terminal_cost(Xnew[t])

        return Xnew, Unew, costnew

    def ilqr(self, x0, U):
        """use iterative linear quadratic regulation to find a control
        sequence that minimizes the cost function

        x0 np.array: the initial state of the system
        U np.array: the initial control trajectory dimensions = [dof, time]
        """
        lamb = 1.0  # regularization parameter

        # simulate forward using the current control trajectory
        X, cost = self.simulate(x0, U)

        for i in range(self.max_iter):

            print("it: {}, cost: {}".format(i, cost))

            derivatives = self.forward_pass(X, U)

            gains = self.backward_pass(lamb, *derivatives)

            Xnew, Unew, costnew = self.calculate_step(X, U, *gains)

            # Levenberg-Marquardt heuristic
            if costnew < cost:

                # decrease lambda (get closer to Newton's method)
                lamb /= self.lamb_factor

                X, U = Xnew, Unew  # update trajectory and control signal

                improvement = abs(cost - costnew) / costnew
                cost = costnew

                # check to see if update is small enough to exit
                if i > 0 and improvement < self.eps_converge:
                    print("Converged!")
                    break
            else:
                # increase lambda (get closer to gradient descent)
                lamb *= self.lamb_factor

                if lamb > self.lamb_max:
                    print("Lambda exceeded max value")
                    break

        print("Finished at iteration = %d; Cost = %.4f; logLambda = %.1f" % (i, costnew, np.log(lamb)))

        return X, U, cost
