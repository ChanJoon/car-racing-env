from numpy import array, zeros, hstack, sin, cos, arctan2
from .acados_ocp_pp import acados_ocp_pp
from .acados_solver import acados_solver

import matplotlib.pyplot as plt


class MPC:

    def __init__(self, vehicle_model, track, config, control_init=None):

        if control_init is None:
            self.control = zeros((2,))
        else:
            self.control = control_init
        acados_ocp = acados_ocp_pp(vehicle_model, track, config)
        self.__solver = acados_solver(acados_ocp, config)

        self.track = track
        self.N = config.mpc.N
        self.dt = config.mpc.dt
        self.nu = acados_ocp.model.u.size1()
        self.nx = acados_ocp.model.x.size1()

        self.pp_cost_mode = config.mpc.pp_cost_mode
        self.pp_dthetaref = config.mpc.pp_ref_horizon_length / self.N

        self.lbu  = array([config.mpc.delta_min,  config.mpc.D_min ])
        self.ubu  = array([config.mpc.delta_max,  config.mpc.D_max ])
        self.lbdu = array([config.mpc.ddelta_min, config.mpc.dD_min])
        self.ubdu = array([config.mpc.ddelta_max, config.mpc.dD_max])

        self.trajectory = None
        self.theta = None


    def solve(self, state, action):

        theta0, ey0 = self.track.get_theta(*state[:2], initial_guess=self.theta, eval_ey=True)
        psiref = arctan2(*self.track(theta0, 1)[::-1])
        
        epsi = state[2] - psiref
        epsi = arctan2(sin(epsi), cos(epsi))

        # Set initial state constraint
        x0 = hstack([theta0, ey0, epsi, state[3:]])
        self.__solver.set(0, "lbx", x0)
        self.__solver.set(0, "ubx", x0)

        if self.theta is None:
            for k in range(self.N):
                self.__solver.set(k, "x", x0)

        if self.pp_cost_mode==0:
            for k in range(self.N):
                self.__solver.set( k, "p", hstack([theta0+k*action[0]/self.N, action[1:]]))
            self.__solver.set(self.N, "p", hstack([theta0+action[0], action[1:]]))
        else:
            for k in range(self.N):
                self.__solver.set( k, "p", hstack([theta0, action]))
            self.__solver.set(self.N, "p", hstack([theta0, action]))

        status = self.__solver.solve()

        x0 = self.__solver.get(0, "x")
        u0 = self.__solver.get(0, "u")
        if self.trajectory is None:
            self.trajectory = zeros((self.N, self.nx))
        self.control = u0

        self.dyn_state = x0 #MEMO [theta, ec, epsi, vx, vy, omega, delta, D]

        for k in range(self.N):
            self.trajectory[k, :] = self.__solver.get(k, "x")
        XY = self.track(self.trajectory[:, 0])
        dXY = self.track(self.trajectory[:, 0], 1)
        psiref = arctan2(dXY[:, 1], dXY[:, 0])
        eX = -sin(psiref) * self.trajectory[:, 1]
        eY =  cos(psiref) * self.trajectory[:, 1]
        self.trajectory[:, 0]  = XY[:, 0] + eX
        self.trajectory[:, 1]  = XY[:, 1] + eY
        self.trajectory[:, 2] += psiref
        if status==0:
            self.theta = self.__solver.get(1, "x")[0]


        return status
    

    def reset_theta(self):
        self.theta = None
