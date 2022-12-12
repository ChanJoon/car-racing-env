from env import Racing, Config
from env.vehicle_model import VehicleModel

from numpy import array, zeros, hstack, sin, cos, arctan2, argmin
import numpy as np

from scipy.optimize import minimize



if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    env = Racing(config, render_mode="human")
    
    state = env.reset()
    for i in range(2000):
        state, reward, done, info = env.step([config.mpc.pp_ref_horizon_length, config.mpc.qtheta, config.mpc.qec])

        if done == True:
            break

