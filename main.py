from env import Racing, Config
from env.vehicle_model import VehicleModel

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    env = Racing(config, render_mode="human")
    
    state = env.reset()

    total_state = np.empty((2000,10))
    for i in range(2000):
        state, reward, done, info = env.step([config.mpc.pp_ref_horizon_length, config.mpc.qtheta, config.mpc.qec])

        total_state[i, :] = np.concatenate((state,info))
        if done == True:
            t = np.linspace(0, i*0.02, i)
            plt.figure()
            plt.subplot(2,1,1)
            plt.step(t, total_state[:i,8], color='r')
            plt.step(t, total_state[:i,9], color='g')
            plt.title('closed-loop simulation')
            plt.legend(['dD','ddelta'])
            plt.ylabel('u')
            plt.xlabel('t')
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.plot(t, total_state[:i,:])
            plt.ylabel('x')
            plt.xlabel('t')
            plt.legend(['x','y','psi','vx','vy','delta','D','ddelta','dD'])
            plt.grid(True)
            plt.show()
            break

