from numpy import array, zeros, zeros_like, hstack, sin, cos, arctan2, clip, linspace
from .acados_ocp import acados_ocp
from .acados_solver import acados_solver
from .bezier_reference import BezierReference


class MPC:

    def __init__(self, vehicle_model, config, control_init=None):

        if control_init is None:
            self.control = zeros((2,))
        else:
            self.control = control_init
        ocp = acados_ocp(vehicle_model, config)
        self.__solver = acados_solver(ocp, config)

        self.N = config.mpc.N
        self.dt = config.mpc.dt
        self.nu = ocp.model.u.size1()
        self.nx = ocp.model.x.size1()

        self.B = BezierReference(self.dt, self.N)

        #self.lbu  = array([config.mpc.D_min, config.mpc.ddelta_min])
        #self.ubu  = array([config.mpc.D_max, config.mpc.ddelta_max])

        self.reference = None
        self.trajectory = None


    def __call__(self, state, action):
        
        # Generate a reference trajectory using Bezier curve
        self.B.update(state[:3], state[3], action)
        self.reference = hstack([
            self.B.get_reference(), zeros((self.N+1, 4))
        ])

        local_ref = zeros_like(self.reference)

        R = array([
            [cos(state[2]), -sin(state[2])],
            [sin(state[2]),  cos(state[2])]
        ])  # Rotation matrix R(psi0)

        # Transform the reference trajectory to the vehicle local frame.
        local_ref[:, :2] = (self.reference[:, :2] - state[:2]).dot(R)  # [X' Y']^T = R(psi0)[X-X0 Y-Y0]^T
        local_ref[:,  2] = self.reference[:, 2] - state[2]  # psi' = psi - psi0
        local_ref[:,  2] = arctan2(sin(local_ref[:, 2]), cos(local_ref[:, 2]))  # normalize angle 

        # Set initial state constraint
        self.__solver.set(0, "lbx", hstack([zeros(3), state[3:]]))
        self.__solver.set(0, "ubx", hstack([zeros(3), state[3:]]))

        # Set the reference trajectory to the solver interface
        for k in range(self.N):
            self.__solver.set(
                k, "yref", 
                hstack([local_ref[k, :], zeros(self.nu)])
            )
        self.__solver.set(self.N, "yref", local_ref[self.N, :])

        # Solve problem
        status = self.__solver.solve()

        if self.trajectory is None:
            self.trajectory = zeros((self.N+1, self.nx))
        for k in range(self.N+1):
            self.trajectory[k, :] = self.__solver.get(k, "x")
        self.trajectory[:, :2] = self.trajectory[:, :2].dot(R.T)
        self.trajectory[:, :3] += state[:3]

        return self.__solver.get(0, "u")
