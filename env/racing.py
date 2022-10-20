from typing import Callable, Optional, Any

from gym.core import Env
from gym import spaces

import numpy as np
from scipy.integrate import solve_ivp

from .renderer import Renderer
from .track import Track
from .live_plot import LivePlot, Clock
from .vehicle_model import VehicleModel
from .utils import RK4
from .config import Config



class Racing(Env):

    metadata = {
        "render_modes": ["human", "web"],
        "render_fps": 0
    }

    def __init__(self, config):

        self.low = np.array([], dtype=np.float32)
        self.high = np.array([], dtype=np.float32)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.dt = config.env.dt
        self.model = VehicleModel(**config.vehicle.get_dict())
        self.track = Track(config.env.randomize_track)
        self.renderer = Renderer(track=self.track)

        self.lbu = np.array([config.mpc.ddelta_min, config.mpc.dD_min])
        self.ubu = np.array([config.mpc.ddelta_max, config.mpc.dD_max])

        self.integrator_type = config.env.integrator_type
        self.sim_method_num_steps = config.env.sim_method_num_steps

        self.state = np.hstack([self.track.array[0, 1:4], [0.01, 0.0, 0.0, 0.0, 0.0]])

    def step(self, action):
        reward = -1.0
        terminated = bool(
            1.0
        )
        u = action
        if self.integrator_type=="RK4":
            for _ in range(self.sim_method_num_steps):
                self.state = RK4(
                    lambda x: np.array(
                        self.model.f(*x, *np.clip(u, self.lbu, self.ubu))
                    ),
                    self.dt/self.sim_method_num_steps, self.state
                )
        else:
            sol = solve_ivp(
                lambda t, x, u: self.model.f(*x, *u),
                (0, self.dt),
                self.state,
                args=(np.clip(u, self.lbu, self.ubu),),
                method=self.integrator_type
            )
            self.state = sol.y[:, -1]

        self.renderer.render_step(self.state)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}


    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # super().reset(seed=seed)
        self.state = np.hstack([self.track.array[0, 1:4], [0.01, 0.0, 0.0, 0.0, 0.0]])  # TODO: Initialization method
        self.renderer.reset()
        self.renderer.render_step(self.state)
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


    def render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        # self.renderer.step(mode)
