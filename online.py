from env import Racing, Config
from env.vehicle_model import VehicleModel
from time import sleep

if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    env = Racing(config, render_mode="online")
    
    state = env.reset()

    print("Game starts in 5 seconds..")
    sleep(5)

    env.start()
    
    for i in range(2000):
        state, reward, done, info = env.step([config.mpc.pp_ref_horizon_length, config.mpc.qtheta, config.mpc.qec])

        if done == True:
            break

