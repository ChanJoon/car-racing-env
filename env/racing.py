from typing import Callable, Optional, Any

from gym.core import Env
from gym import spaces

import numpy as np
from scipy.integrate import solve_ivp

from .renderer import Renderer
from .track import Track
from .vehicle_model import VehicleModel
from .utils import RK4
from .config import Config

import copy

class Racing(Env):

    metadata = {
        "render_modes": ["human", "web"],
        "render_fps": 0
    }

    def __init__(self, config, render_mode='human'):

        self.low = np.array([], dtype=np.float32)
        self.high = np.array([], dtype=np.float32)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.dt = config.env.dt
        self.max_laps = config.env.max_laps
        self.model = VehicleModel(**config.vehicle.get_dict())
        self.track = Track(config.env.track_row, config.env.track_col, config.env.randomize_track)
        self.renderer = Renderer("", "", track=self.track, model=self.model)

        self.lbu = np.array([config.vehicle.ddelta_min, config.vehicle.dD_min])
        self.ubu = np.array([config.vehicle.ddelta_max, config.vehicle.dD_max])

        self.integrator_type = config.env.integrator_type
        self.sim_method_num_steps = config.env.sim_method_num_steps

        self.state = np.hstack([[2, 0, 0], [0.01, 0.0, 0.0, 0.0, 0.0]])

        self.lap_count = -1
        self.last_lap = 0
        self.temp_lap_flag = False

    def step(self, action):
        reward = -1.0
        
        done = False
        u = action
        sol = solve_ivp(
                lambda t, x, u: self.model.f(*x, *u),
                (0, self.dt),
                self.state,
                args=(np.clip(u, self.lbu, self.ubu),),
                method=self.integrator_type
            )
        self.state = sol.y[:, -1]

        # 시작선 지남
        if (2.4 < self.x <= 2.5) and (-0.5 < self.y < 0.5) and self.temp_lap_flag == False:
            self.lap_count += 1
            self.temp_lap_flag = True

        if (2.5 < self.x) and (-0.5 < self.y < 0.5) and self.temp_lap_flag == True:
            self.temp_lap_flag = False

        if self.last_lap < self.lap_count:
            # 한바퀴를 돌았다.
            self.last_lap = self.lap_count
            if self.lap_count == self.max_laps:
                done = True

        self.renderer.render_step(self.state)
        return np.array(self.state, dtype=np.float32), reward, done, {}


    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # if self.track.randomize_track == True or self.track.disc_coords is None:
        self.track.createTrack()
        # super().reset(seed=seed)
        self.state = np.hstack([[2, 0, 0], [0.01, 0.0, 0.0, 0.0, 0.0]])  # TODO: Initialization method
        self.renderer.reset()
        self.renderer.show()

        
        
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        assert mode in self.metadata["render_modes"]
        # self.renderer.step(mode)

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def phi(self):
        return self.state[2]

    @property
    def vx(self):
        return self.state[3]

    @property
    def vy(self):
        return self.state[4]

    @property
    def vw(self):
        return self.state[5]

    @property
    def e(self):
        return self.track.getCenterLineError(self.x, self.y)

