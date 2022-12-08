import numpy as np
import redis
from collections import defaultdict
from scipy.integrate import solve_ivp
from env.vehicle_model import VehicleModel
import multiprocessing as mp

dt = 0.02

lbu = np.array([-1e0, -1e0])
ubu = np.array([1e0, 1e0])

class RedisRacingServer():
    def __init__(self):
        self.rd = redis.StrictRedis(host='localhost', port=6379, db=0)
        self.rooms = defaultdict(list)
        self.room_infos = defaultdict(lambda: {
            "status": "stand_by",
            "room_t": 0,
        })

        self.room_id = None
        self.username = None

        self.vehicle_model = VehicleModel()

    def join(self, room_id, username):
        if not username in self.rooms[room_id] and self.room_infos[room_id]["status"] == "stand_by":
            self.rooms[room_id].append(username)

        self.rd.set(self.rd_key("room_usernames", room_id, None), "_".join(self.rooms[room_id]))

        room_t = self.rd.get(self.rd_key("room_t", room_id, username))
        if room_t is None:
            self.rd.set(self.rd_key("room_t", room_id, username), self.room_infos[room_id]["room_t"])

        return True

    def start(self, room_id):
        self.rd.set(self.rd_key('room_status', room_id, None), "racing")

        return self.room_info[room_id]

    def leave(self, room_id, username):
        if username in self.rooms[room_id]:
            self.rooms[room_id].remove(username)

        self.rd.set(self.rd_key("room_usernames", room_id, None), "_".join(self.rooms[room_id]))

    def room_info(self, room_id):
        room_status = self.rd.get(self.rd_key('room_status', room_id, None))
        if room_status is None:
            room_status = "stand_by"
            self.rd.set(self.rd_key('room_status', room_id, None), room_status)
        try:
            self.room_infos[room_id]["status"] = room_status.decode('UTF-8')
        except:
            self.room_infos[room_id]["status"] = room_status

        return self.room_infos[room_id]

    def step(self):
        # solving joins
        while True:
            d = self.rd.rpop(self.rd_key('join_queue', None, None))
            if d is None:
                break
            
            d = d.decode('utf-8')
            print("JOIN", d.split('_'))
            # room_id + _ + username
            if len(d.split('_')) == 2:
                
                self.join(d.split('_')[0], d.split('_')[1])

        # solve the model
        for room_id in self.rooms:
            usernames = self.rooms[room_id]
            if self.room_info(room_id)["status"] != "racing":
                continue

            # num_cores = mp.cpu_count()
            # pool = mp.Pool(num_cores)            

            # pool.map(self.compute, usernames)
            # pool.close()
            # pool.join()

            for username in usernames:
                self.compute(username, room_id)

            self.setRoomT(room_id)

    def compute(self, username, room_id):
        print(username)
        user_t = int(self.rd.get(self.rd_key("step", room_id, username)))
        server_t = int(self.rd.get(self.rd_key("step_server", room_id, username)))
        
        print(user_t, server_t)
        while server_t < user_t:
            state_str = self.rd.get(self.rd_key("state", room_id, username)).decode('UTF-8')
            u_str = self.rd.get(self.rd_key("action", room_id, username)).decode('UTF-8')

            state = eval('np.array('+state_str+')')
            u     = eval('np.array('+  u_str  +')')

            # 계산..
            sol = solve_ivp(
                lambda t, x, u: self.vehicle_model.f(*x, *u),
                (0, dt),
                state,
                args=(np.clip(u, lbu, ubu),),
                method="RK45"
            )
            state = sol.y[:, -1]

            state_str = np.array2string(state, separator=',', suppress_small=True)
            self.rd.set(self.rd_key("state_server", room_id, username), state_str)

            server_t = int(self.rd.incr(self.rd_key("step_server", room_id, username)))    
                
    def room_t(self, room_id):
        user_ts = []
        for username in self.rooms[room_id]:
            user_t = int(self.rd.get(self.rd_key("step", room_id, username)))
            user_ts.append(user_t)

        if len(user_ts) > 0:
            min_t = min(user_ts)
            self.room_infos[room_id]["room_t"] = min_t
            return min_t
        else:
            return 0

    def setRoomT(self, room_id):
        self.rd.set(room_id+"_t", self.room_t(room_id))

    def rd_key(self, k, room_id, username):
        if k == "step":
            return room_id+"_"+username+"_step"
        if k == "step_server":
            return room_id+"_"+username+"_step_server"
        if k == "state":
            return room_id+"_"+username+"_state"
        if k == "state_server":
            return room_id+"_"+username+"_state_server"
        if k == "action":
            return room_id+"_"+username+"_action"
        if k == "room_status":
            return room_id+"_room_status"
        if k == "room_t":
            return room_id+"_t"
        if k == "room_usernames":
            return room_id+"_usernames"
        if k == "join_queue":
            return "join_queue"
        if k == "leave_queue":
            return "leave_queue"

if __name__ == '__main__':
    ns = RedisRacingServer()
    while True:
        ns.step()

