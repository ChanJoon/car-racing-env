import casadi as ca
from numpy import array, zeros, ones, hstack, diag, eye
from scipy.linalg import block_diag
from acados_template import AcadosOcp
from env.utils import RK4



def acados_ocp(vehicle_model, config):

    ocp = AcadosOcp()

    ## CasADi Model

    # set up states & controls
    X     = ca.MX.sym("X")
    Y     = ca.MX.sym("Y")
    psi   = ca.MX.sym("psi")
    vx    = ca.MX.sym("vx")
    vy    = ca.MX.sym("vy")
    omega = ca.MX.sym("omega")
    delta = ca.MX.sym("delta")
    D     = ca.MX.sym("D")
    x = ca.vertcat(X, Y, psi, vx, vy, omega, delta, D)
    ocp.model.x = x
    nx = ocp.model.x.size1()

    # controls
    ddelta = ca.MX.sym("ddelta")
    dD = ca.MX.sym("dD")
    u = ca.vertcat(ddelta, dD)
    ocp.model.u = u
    nu = ocp.model.u.size1()

    ny = nx + nu
    ny_e = nx

    # xdot
    Xdot     = ca.MX.sym("Xdot")
    Ydot     = ca.MX.sym("Ydot")
    psidot   = ca.MX.sym("psidot")
    vxdot    = ca.MX.sym("vxdot")
    vydot    = ca.MX.sym("vydot")
    omegadot = ca.MX.sym("omegadot")
    deltadot = ca.MX.sym("deltadot")
    Ddot     = ca.MX.sym("Ddot")
    xdot = ca.vertcat(Xdot, Ydot, psidot, vxdot, vydot, omegadot, deltadot, Ddot)
    ocp.model.xdot = xdot

    # algebraic variables
    z = ca.vertcat([])
    ocp.model.z = z
    nz = ocp.model.z.size1()

    # parameters
    p = ca.vertcat([])
    ocp.model.p = p
    np = ocp.model.p.size1()
    ocp.parameter_values = zeros(np)

    # dynamics
    f_expl = ca.vertcat(
        *vehicle_model.f(
            X, Y, psi, vx, vy, omega,
            delta, D, ddelta, dD
        )
    )
    xnew = [x]
    for _ in range(config.mpc.sim_method_num_steps):
        xnew.append(
            RK4(
                lambda xk: ca.vertcat(
                    *vehicle_model.f(
                        xk[0], xk[1], xk[2], xk[3], xk[4], xk[5],
                        delta, D, ddelta, dD
                    )
                ),
                config.mpc.dt/config.mpc.sim_method_num_steps, xnew[-1]
            )
        )
    disc_dyn = xnew[-1]

    ocp.model.f_impl_expr = xdot - f_expl
    ocp.model.f_expl_expr = f_expl
    ocp.model.disc_dyn_expr = disc_dyn

    ocp.model.con_h_expr = ca.vertcat([])
    nh = ocp.model.con_h_expr.size1()
    nsh = nh

    ocp.model.name = "Path_parametric_MPC"


    # set constraints

    ocp.constraints.x0 = zeros(nx)

    ocp.constraints.lbx = array([config.mpc.delta_min, config.mpc.D_min])
    ocp.constraints.ubx = array([config.mpc.delta_max, config.mpc.D_max])
    ocp.constraints.idxbx = array([nx-2, nx-1])
    nsbx = ocp.constraints.idxbx.shape[0]
    ocp.constraints.lbu = array([config.mpc.ddelta_min, config.mpc.dD_min])
    ocp.constraints.ubu = array([config.mpc.ddelta_max, config.mpc.dD_max])
    ocp.constraints.idxbu = array(range(nu))

    ocp.constraints.lsbx = zeros([nsbx])
    ocp.constraints.usbx = zeros([nsbx])
    ocp.constraints.idxsbx = array(range(nsbx))

    ocp.constraints.lh = array([])
    ocp.constraints.uh = array([])
    ocp.constraints.lsh = zeros(nsh)
    ocp.constraints.ush = zeros(nsh)
    ocp.constraints.idxsh = array(range(nsh))


    # set cost

    Q = diag(config.mpc.Q)

    R = diag(config.mpc.R)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    Vx = zeros((ny, nx))
    Vx[:nx, :nx] = eye(nx)
    ocp.cost.Vx = Vx

    Vu = zeros((ny, nu))
    Vu[nx:, :] = eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = zeros((ny_e, nx))
    Vx_e[:nx, :nx] = eye(nx)
    ocp.cost.Vx_e = Vx_e

    unscale = 1 / config.mpc.dt
    ocp.cost.W = unscale * block_diag(Q, R)
    ocp.cost.W_e = Q

    # set intial references
    ocp.cost.yref = zeros(ny)
    ocp.cost.yref_e = zeros(ny_e)


    ns = nsh + nsbx
    ocp.cost.zl = 100 * ones((ns,))
    ocp.cost.zu = 100 * ones((ns,))
    ocp.cost.Zl = 1 * ones((ns,))
    ocp.cost.Zu = 1 * ones((ns,))


    return ocp