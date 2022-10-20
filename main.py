from env import Racing, Config
from env.vehicle_model import VehicleModel

if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    env = Racing(config, render_mode="human")

    state = env.reset()
    for i in range(10):
        action = [0.1, 0.1]
        state, _, _, _, _ = env.step(action)

        print(state)

    env.render()
