'''
Racing environment configuration file'''
from .utils import Parameters, dataclass
from dataclasses import field



@dataclass
class Environment_Parameters(Parameters):

    integrator_type : str = "RK45"  # ["RK4", "RK45", "RK23", "DOP853", "radau", "BDF", "LSODA"]
    dt : float = 0.02
    sim_method_num_steps : int = 3
    randomize_track : bool = True
    track_row : int = 5
    track_col : int = 7
    is_constant_track_border : bool = True
    render_fps : int = 60  # Set 0 if fps limit to be disabled.
    max_laps : int = 1  # 한 에피소드에 돌 랩 수

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


@dataclass
class Config(Parameters):

    env : Environment_Parameters = Environment_Parameters()
    vehicle : Vehicle_Parameters = Vehicle_Parameters()
