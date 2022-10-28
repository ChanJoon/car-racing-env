from env import Racing, Config
from env.vehicle_model import VehicleModel

def PID(Kp, Ki, Kd, MV_bar=0):
    # initialize stored data
    e_prev = 0
    t_prev = -1
    I = 0
    
    # initial control
    MV = MV_bar
    
    while True:
        # yield MV, wait for new t, PV, SP
        t, PV, SP = yield MV
        
        # PID calculations
        e = SP - PV
        
        P = Kp*e
        I = I + Ki*e*(t - t_prev)
        D = Kd*(e - e_prev)/(t - t_prev)
        
        MV = MV_bar + P + I + D
        
        # update stored data for next iteration
        e_prev = e
        t_prev = t

if __name__ == "__main__":

    config = Config()

    vehicle_model = VehicleModel(**config.vehicle.get_dict())

    env = Racing(config, render_mode="human")

    controller_theta = PID(0.1, 1, 10)        # create pid control for steering
    controller_theta.send(None)    

    controller_throttle = PID(2, 0.1, 1) 
    controller_throttle.send(None)

    state = env.reset()
    for i in range(2000):
        u1 = controller_theta.send([i / 100, env.e, 0])
        u2 = controller_throttle.send([i / 100, env.vx, 0.5])

        action = [u1, u2]
        state, reward, done, info = env.step(action)

        if done == True:
            break

