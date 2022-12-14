'''
Racing environment configuration file'''
from .utils import Parameters, dataclass
from dataclasses import field



@dataclass
class Environment_Parameters(Parameters):

    integrator_type : str = "MPC"  # ["RK4", "RK45", "RK23", "DOP853", "radau", "BDF", "LSODA"]
    dt : float = 0.02
    sim_method_num_steps : int = 3
    randomize_track : bool = True
    track_row : int = 5
    track_col : int = 7
    is_constant_track_border : bool = True
    render_fps : int = 60  # Set 0 if fps limit to be disabled.
    max_laps : int = 2  # 한 에피소드에 돌 랩 수

    server_path : str = "172.24.203.250"
    username : str = "chanjoon"
    room_id : str = "test"

@dataclass
class Vehicle_Parameters(Parameters):

    m   : float = 0.041
    Iz  : float = 2.78e-5
    lf  : float = 0.029
    lr  : float = 0.033
    Cm1 : float = 0.287
    Cm2 : float = 0.0545
    Cr0 : float = 0.0518
    Cr2 : float = 3.5e-4
    Cr3 : float = 5.0
    Br  : float = 3.3852
    Cr  : float = 1.2691
    Dr  : float = 0.1737
    Bf  : float = 2.579
    Cf  : float = 1.2
    Df  : float = 0.192
    L   : float = 0.12
    W   : float = 0.06

    ddelta_min : float = -1e0
    ddelta_max : float =  1e0
    dD_min : float = -1e0
    dD_max : float =  1e0

    delta_min : float = -0.8 # ~= 45deg
    delta_max : float =  0.8
    D_min : float = -1.
    D_max : float =  1.

@dataclass
class MPC_Parameters(Parameters):

    dt : float = 0.02
    N  : int   = 50
    delta_min : float = -0.8
    delta_max : float =  0.8
    D_min : float = -1.
    D_max : float =  1.

    ddelta_min : float = -1e0
    ddelta_max : float =  1e0
    dD_min : float = -1e0
    dD_max : float =  1e0

    Q : list = field(default_factory=lambda:
        [ 1e-1, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3 ]
    )
    R : list = field(default_factory=lambda:
        [1e-3, 5e-3]
    )

    integrator_type : str = "ERK"  # ["ERK", "IRK", "DISCRETE"]
    sim_method_num_stages : int = 3
    sim_method_num_steps  : int = 3
    qp_solver : str = "PARTIAL_CONDENSING_HPIPM"
    qp_solver_warm_start : int = 1
    qp_solver_iter_max   : int = 50
    hpipm_mode : str = "BALANCE"
    hessian_approx  : str = "GAUSS_NEWTON"
    nlp_solver_type : str = "SQP_RTI"
    nlp_solver_max_iter : int = 200
    tol : float = 1e-4
    print_level : int = 0

    pp_cost_mode : int = 0  # [0: min((thetaref - theta)^2), 1: min(theta0-theta), 2: min(1/(theta-theta0))]
    pp_ref_horizon_length : float = 1.6
    qtheta = 1e-1
    qec = 1e-8

@dataclass
class Config(Parameters):

    env : Environment_Parameters = Environment_Parameters()
    vehicle : Vehicle_Parameters = Vehicle_Parameters()
    mpc : MPC_Parameters = MPC_Parameters()
