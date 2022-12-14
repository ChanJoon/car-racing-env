import casadi as ca
from numpy import array, zeros, ones, hstack, diag, eye
from scipy.linalg import block_diag
from acados_template import AcadosOcp
from ..utils import RK4



def acados_ocp_pp(vehicle_model, track, config):

    ocp = AcadosOcp()

    ## CasADi Model

    kapparef = ca.interpolant("kapparef_s", "bspline", [track.thetaref], track.kapparef)

    # set up states & controls
    theta = ca.MX.sym("theta")
    ec = ca.MX.sym("ec")
    epsi = ca.MX.sym("epsi")
    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    omega = ca.MX.sym("omega")
    delta = ca.MX.sym("delta")
    D = ca.MX.sym("D")
    x = ca.vertcat(theta, ec, epsi, vx, vy, omega, delta, D)
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
    thetadot = ca.MX.sym("thetadot")
    ecdot = ca.MX.sym("ecdot")
    epsidot = ca.MX.sym("epsidot")
    vxdot = ca.MX.sym("vxdot")
    vydot = ca.MX.sym("vydot")
    omegadot = ca.MX.sym("omegadot")
    deltadot = ca.MX.sym("deltadot")
    Ddot = ca.MX.sym("Ddot")
    xdot = ca.vertcat(thetadot, ecdot, epsidot, vxdot, vydot, omegadot, deltadot, Ddot)
    ocp.model.xdot = xdot

    # algebraic variables
    z = ca.vertcat([])
    ocp.model.z = z
    nz = ocp.model.z.size1()

    # parameters
    thetaref = ca.MX.sym("thetaref")
    qtheta = ca.MX.sym("qtheta")
    qec = ca.MX.sym("qec")
    p = ca.vertcat(thetaref, qtheta, qec)
    ocp.model.p = p
    np = ocp.model.p.size1()
    ocp.parameter_values = zeros(np)

    # dynamics
    f_expl = ca.vertcat(
        *vehicle_model.f_pp(
            theta, ec, epsi, vx, vy, omega,
            delta, D, ddelta, dD,
            lambda thetak: kapparef(ca.mod(thetak, track.thetaref[-1]))
        )
    )
    xnew = [x]
    for _ in range(config.mpc.sim_method_num_steps):
        xnew.append(
            RK4(
                lambda xk: ca.vertcat(
                    *vehicle_model.f_pp(
                        xk[0], xk[1], xk[2], xk[3], xk[4], xk[5],
                        delta, D, ddelta, dD,
                        lambda thetak: kapparef(ca.mod(thetak, track.thetaref[-1]))
                    )
                ),
                config.mpc.dt/config.mpc.sim_method_num_steps, xnew[-1]
            )
        )
    disc_dyn = xnew[-1]

    ocp.model.f_impl_expr = xdot - f_expl
    ocp.model.f_expl_expr = f_expl
    ocp.model.disc_dyn_expr = disc_dyn

    if config.env.is_constant_track_border:
        ocp.model.con_h_expr = ca.vertcat(ec, delta, D)
    else:
        border_l = ca.interpolant("bound_l_s", "bspline", [track.thetaref], track.border_left)
        border_r = ca.interpolant("bound_r_s", "bspline", [track.thetaref], track.border_right)
        ubec = border_l(theta) - vehicle_model.safe_distance
        lbec = border_r(theta) + vehicle_model.safe_distance
        normalized_ec = (ec - ubec + lbec) * 2 / (ubec + lbec)

        ocp.model.con_h_expr = ca.vertcat(normalized_ec, delta, D)
    nh = ocp.model.con_h_expr.size1()
    nsh = nh

    if config.mpc.pp_cost_mode==0:
        ocp.model.cost_expr_ext_cost = qtheta*(thetaref-theta)**2 + qec*ec*ec + 1e-3*D*D + 5e-3*delta*delta + 1e-3*dD*dD + 5e-3*ddelta*ddelta
        ocp.model.cost_expr_ext_cost_e = qtheta*(thetaref-theta)**2 + qec*ec*ec + 1e-3*D*D + 5e-3*delta*delta
    else:
        ocp.model.cost_expr_ext_cost = qec*ec*ec + 1e-3*D*D + 5e-3*delta*delta + 1e-3*dD*dD + 5e-3*ddelta*ddelta
        if config.mpc.pp_cost_mode==1:
            ocp.model.cost_expr_ext_cost_e = qtheta*(thetaref-theta) + qec*ec*ec + 1e-3*D*D + 5e-3*delta*delta
        elif config.mpc.pp_cost_mode==2:
            ocp.model.cost_expr_ext_cost_e = qtheta/(1+theta-thetaref) + qec*ec*ec + 1e-3*D*D + 5e-3*delta*delta

    ocp.model.name = "Path_parametric_MPC"


    # set constraints

    ocp.constraints.x0 = zeros(nx)

    ocp.constraints.lbx = array([-12])
    ocp.constraints.ubx = array([12])
    ocp.constraints.idxbx = array([1])
    nsbx = ocp.constraints.idxbx.shape[0]
    ocp.constraints.lbu = array([config.mpc.ddelta_min, config.mpc.dD_min])
    ocp.constraints.ubu = array([config.mpc.ddelta_max, config.mpc.dD_max])
    ocp.constraints.idxbu = array(range(nu))

    ocp.constraints.lsbx = zeros([nsbx])
    ocp.constraints.usbx = zeros([nsbx])
    ocp.constraints.idxsbx = array(range(nsbx))

    if config.env.is_constant_track_border:
        ocp.constraints.lh = array([-track.border_right[0] + vehicle_model.safe_distance, config.mpc.delta_min, config.mpc.D_min])
        ocp.constraints.uh = array([  track.border_left[0] - vehicle_model.safe_distance, config.mpc.delta_max, config.mpc.D_max])
    else:
        ocp.constraints.lh = array([-1, config.mpc.delta_min, config.mpc.D_min])
        ocp.constraints.uh = array([ 1, config.mpc.delta_max, config.mpc.D_max])
    ocp.constraints.lsh = zeros(nsh)
    ocp.constraints.ush = zeros(nsh)
    ocp.constraints.idxsh = array(range(nsh))


    # set cost

    Q = diag(config.mpc.Q)

    R = diag(config.mpc.R)

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ns = nsh + nsbx
    ocp.cost.zl = 100 * ones((ns,))
    ocp.cost.zu = 100 * ones((ns,))
    ocp.cost.Zl = 1 * ones((ns,))
    ocp.cost.Zu = 1 * ones((ns,))


    return ocp