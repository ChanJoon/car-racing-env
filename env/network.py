import time
import numpy as np

class Network():
    def __init__(self) -> None:
        pass

    def joinRoom(self):
        self.rd.lpush(self.rd_key('join_queue'), self.room_id+"_"+self.username)

        while True:
            room_usernames = self.rd.get(self.rd_key('room_usernames'))
            if room_usernames is None:
                time.sleep(0.1)
                continue
            
            room_usernames = room_usernames.decode('UTF-8')
            if self.username in room_usernames.split('_'):
                print("Joined", self.room_id, self.username)
                return True
            else:
                time.sleep(0.1)

    def start(self):
        self.rd.set(self.rd_key('room_status'), "racing")

        return True

    def leaveRoom(self):
        self.rd.lpush(self.rd_key('leave_queue'), self.room_id+"_"+self.username)

        while True:
            room_usernames = self.rd.get(self.rd_key('room_usernames'))
            if room_usernames is None:
                time.sleep(0.1)
            
            room_usernames = room_usernames.decode('UTF-8')
            if self.username not in room_usernames.split('_'):
                print("Left", self.room_id, self.username)
                return True
            else:
                time.sleep(0.1)

    def resetOnlineSettings(self):
        self.t = 0

        self.rd.set(self.rd_key("step"), self.t)
        self.rd.set(self.rd_key("step_server"), self.t)

    def getCarStates(self):
        usernames = self.rd.get(self.rd_key("room_usernames"))

        if usernames is not None:
            usernames = usernames.decode('UTF-8').split("_")
        else:
            usernames = []
        
        for username in usernames:
            state_str = self.rd.get(self.room_id+"_"+username+"_state_server").decode('UTF-8')

            if state_str is not None:
                self.car_states[username] = eval('np.array(' + state_str + ')')

    def rd_key(self, k):
        if k == "step":
            return self.room_id+"_"+self.username+"_step"
        if k == "step_server":
            return self.room_id+"_"+self.username+"_step_server"
        if k == "state":
            return self.room_id+"_"+self.username+"_state"
        if k == "state_server":
            return self.room_id+"_"+self.username+"_state_server"
        if k == "action":
            return self.room_id+"_"+self.username+"_action"
        if k == "room_status":
            return self.room_id+"_room_status"
        if k == "room_t":
            return self.room_id+"_t"
        if k == "room_usernames":
            return self.room_id+"_usernames"
        if k == "init_pos":
            return self.room_id+"_"+self.username+"_init_pos"
        if k == "join_queue":
            return "join_queue"
        if k == "leave_queue":
            return "leave_queue"