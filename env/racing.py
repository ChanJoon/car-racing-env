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
from .network import Network

import copy
import redis
import time
from collections import defaultdict

class Racing(Env, Network):

    metadata = {
        "render_modes": ["human", "online"],
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

        self.init_x = 2.
        self.init_y = 0.

        self.init_state = np.hstack([[self.init_x, self.init_y, 0], [0.01, 0.0, 0.0, 0.0, 0.0]])
        self.state = self.init_state

        self.lap_count = -1
        self.last_lap = 0
        self.temp_lap_flag = False

        self.render_mode = render_mode
        if self.render_mode == "online":
            self.server_path = config.env.server_path
            self.rd = redis.StrictRedis(host=self.server_path, port=6379, db=0)
            self.username = "".join(config.env.username.split("_"))
            self.room_id = "".join(config.env.room_id.split("_"))

            self.joinRoom()
            self.resetOnlineSettings()

            self.car_states = defaultdict(lambda: self.init_state)

        self.t = 0

    def step(self, action):
        reward = -1.0
        
        done = False
        u = action
        self.state = self.compute(u)

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

        self.getCarStates()
        self.renderer.render_step(self.state, self.render_mode, self.car_states)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def compute(self, u):
        if self.render_mode == 'online':
            # TODO : state 정보 json string으로 저장
            state_str = np.array2string(self.state, separator=',', suppress_small=True)
            u_str     = np.array2string(np.asarray(u), separator=',', suppress_small=True)
            self.rd.set(self.rd_key("state"), state_str)
            self.rd.set(self.rd_key("action"), u_str)

            self.t = int(self.rd.incr(self.rd_key("step")))
            while True:
                if self.rd.get(self.rd_key("room_status")) != b'racing':
                    print(self.rd.get(self.rd_key("room_status")))
                    time.sleep(0.01)
                    continue

                server_t = int(self.rd.get(self.rd_key("step_server")))
                room_t   = int(self.rd.get(self.rd_key("room_t")))
                print(server_t, self.t, room_t)
                if server_t == self.t == room_t:
                    # TODO : json string parse 후 저장
                    state_str = self.rd.get(self.rd_key("state_server")).decode('UTF-8')
                    return eval('np.array(' + state_str + ')')
                else:
                    time.sleep(0.01)
        else:
            sol = solve_ivp(
                    lambda t, x, u: self.model.f(*x, *u),
                    (0, self.dt),
                    self.state,
                    args=(np.clip(u, self.lbu, self.ubu),),
                    method=self.integrator_type
                )
        return sol.y[:, -1]
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        if self.track.randomize_track == True or self.track.disc_coords is None:
            self.track.createTrack()
        # super().reset(seed=seed)
        # 출발 좌표 지정
        if self.render_mode == "online":
            # TODO : init pos 없을 경우 처리 / 서버에 요청
            init_pos = self.rd.get(self.rd_key("init_pos"))
            if init_pos is not None:
                self.init_x = float(init_pos.split(';')[0])
                self.init_y = float(init_pos.split(';')[1])

                self.init_state = np.hstack([[self.init_x, self.init_y, 0], [0.01, 0.0, 0.0, 0.0, 0.0]])
            
        self.state = self.init_state  # TODO: Initialization method
        self.renderer.reset()
        self.renderer.show()

        if self.render_mode == "online":
            self.resetOnlineSettings()
        
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

