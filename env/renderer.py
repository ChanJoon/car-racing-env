import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import Rectangle, gca
try:
    from env.vehicle_model import VehicleModel
    from env.track import Track 
except:
    from vehicle_model import VehicleModel
    from track import Track 
from PIL import Image
import matplotlib.transforms as mtransforms
from matplotlib.figure import figaspect
import matplotlib.animation as animation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.artist import Artist
import matplotlib.patches as patches


# matplotlib.use('TkAgg')
import platform
if platform.system() == 'Darwin':
    matplotlib.use('Qt5Agg')

cwd = os.path.abspath(os.getcwd())
class Renderer:

    def __init__(
        self,
        hostname,
        channel_id,
        track,
        model,
    ):

        self.hostname = hostname
        self.channel_id = channel_id
        self.track = track
        self.model = model
        self.render_list = []

        self.cars = []

        self.i = 0

        self.fig = plt.figure(figsize=figaspect(7/5))
        self.ax = plt.gca()
        plt.axis('off')
        self.fig.canvas.draw()

    def render_step(self, state, render_mode="human", car_states=None) -> None:
        """
        Send state data to server via socket.io
        """
        self.render_list.append(state)

        if render_mode == "human":
            self.update(state)
        elif render_mode == "online":
            self.batchUpdate(car_states)


    def reset(self):
        """Resets the render collection list.
        This method should be usually called inside environment's reset method.
        """
        plt.gca().cla()
        self.render_list = []
        
        ax = self.ax
        ax.set_autoscaley_on(False)
        ax.set_autoscalex_on(False)
        ax.set_ylim([-1,5])
        ax.set_xlim([-1,6])

        # Start Sections
        imageData = plt.imread(cwd+'/env/images/start_1.png')
        plt.imshow(imageData, extent=(1.5, 2.5, -0.5, 0.5))

        imageData = plt.imread(cwd+'/env/images/start_2.png')
        plt.imshow(imageData, extent=(0.5, 1.5, -0.5, 0.5))

        imageData = plt.imread(cwd+'/env/images/straight_hor.png')
        plt.imshow(imageData, extent=(2.5, 3.5, -0.5, 0.5))

        imageData = plt.imread(cwd+'/env/images/curve_3.png')
        plt.imshow(imageData, extent=(-0.5, 0.5, -0.5, 0.5))

        self.drawByActions(self.track.actions)
        plt.draw()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.car_boxes = [1]

        # self.car_boxes.append(patches.Rectangle((0, 0), 0, 0, fc='y'))
        

        imageData = Image.open(cwd+'/env/images/cars/car_6.png')
        imageData = imageData.rotate(90, expand=True)


        imageData = OffsetImage(imageData, zoom=0.12)

        # box = AnnotationBbox(imageData, (2, 0), xycoords='data', frameon=False)
        # # point = self.ax.scatter(2, 0)
        # self.ax.draw_artist(box)

        # self.fig.canvas.start_event_loop(0.001)
        # self.car_boxes.append(box)
        

    def drawByActions(self, actions):
        x = 4
        y = 0
        last_action = actions[0]

        # y+=1 # First Action is N

        for idx, action in enumerate(actions):
            if (idx == 0 or idx == len(actions) - 2 or idx == len(actions) - 1):
                continue
            
            if (last_action == "N" and action == "N"):
                imageData = plt.imread(cwd+"/env/images/straight_ver.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += 1
            elif (last_action == "N" and action == "E"):
                imageData = plt.imread(cwd+"/env/images/curve_2.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += 1
            elif (last_action == "N" and action == "W"):
                imageData = plt.imread(cwd+"/env/images/curve_1.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += -1
            elif (last_action == "E" and action == "N"):
                imageData = plt.imread(cwd+"/env/images/curve_4.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += 1
            elif (last_action == "E" and action == "E"):
                imageData = plt.imread(cwd+"/env/images/straight_hor.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += 1
            elif (last_action == "E" and action == "S"):
                imageData = plt.imread(cwd+"/env/images/curve_1.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += -1
            elif (last_action == "S" and action == "E"):
                imageData = plt.imread(cwd+"/env/images/curve_3.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += 1
            elif (last_action == "S" and action == "S"):
                imageData = plt.imread(cwd+"/env/images/straight_ver.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += -1
            elif (last_action == "S" and action == "W"):
                imageData = plt.imread(cwd+"/env/images/curve_4.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += -1
            elif (last_action == "W" and action == "N"):
                imageData = plt.imread(cwd+"/env/images/curve_3.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += 1
            elif (last_action == "W" and action == "S"):
                imageData = plt.imread(cwd+"/env/images/curve_2.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                y += -1
            elif (last_action == "W" and action == "W"):
                imageData = plt.imread(cwd+"/env/images/straight_hor.png")
                plt.imshow(imageData, extent=(x-0.5, x+0.5, y-0.5, y+0.5))
                x += -1
            
            last_action = action
            
    def update(self, state):
        self.i += 1
        # Car 
        self.fig.canvas.restore_region(self.background)
        for car_box in self.car_boxes:
            

            vx, vy = self.model.footprint(state[0], state[1], state[2])

            self.ax.draw_artist(
                self.ax.plot(np.hstack((vx, vx[:1])), np.hstack((vy, vy[:1])), color='k', linewidth=1, animated=True)[0]
            )
            pass

            # car_box._angle = -np.rad2deg(yaw[i])
            # car_box.remove()

        # self.fig.canvas.draw()
        # self.car_boxes.append(self.ax.add_artist(box))
        self.fig.canvas.blit(self.ax.bbox)

        self.fig.canvas.start_event_loop(0.001)
    
    def batchUpdate(self, car_states):
        self.i += 1
        # Car 
        self.fig.canvas.restore_region(self.background)
        for username in car_states:
            state = car_states[username]
            print("BATCH", state)
            vx, vy = self.model.footprint(state[0], state[1], state[2])

            self.ax.draw_artist(
                self.ax.plot(np.hstack((vx, vx[:1])), np.hstack((vy, vy[:1])), color='k', linewidth=1, animated=True)[0]
            )

            # car_box._angle = -np.rad2deg(yaw[i])
            # car_box.remove()

        # self.fig.canvas.draw()
        # self.car_boxes.append(self.ax.add_artist(box))
        self.fig.canvas.blit(self.ax.bbox)

        self.fig.canvas.start_event_loop(0.001)

    def show(self):
        self.fig.show()

if __name__ == "__main__":
    track = Track()
    model = VehicleModel()
    # print(track.getCenterLineError(1, 2))
    renderer = Renderer("", "", track, model)
    renderer.reset()
    renderer.show()
    for i in range(20):
        renderer.update()
