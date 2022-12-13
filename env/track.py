import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import random

from numpy import argmin, sin, cos, arctan2
from .FrenetOptimalTrajectory import frenet_optimal_trajectory
from .DynamicWindowApproach import dynamic_window_approach
from .config import Config

class Track:

  def __init__(self, row_num=5, col_num=7, randomize_track=True):
    self.row_num = row_num
    self.col_num = col_num

    self.randomize_track = randomize_track
    # self.FrenetPath = frenet_optimal_trajectory.FrenetPath

  def __call__(self, theta, *args, **kwargs):
      return self.__spline(theta, *args, **kwargs)

  def __iter__(self):
      return iter(self.__data)

  def __getitem__(self, key):
      return self.__data[key]

  def __setitem__(self, key, val):
      self.__data[key] = val
    
  def get_theta(self, X, Y, initial_guess=None, eval_ey=False):
      if initial_guess is None:
          initial_guess = self.thetaref[argmin((self.Xref - X)**2 + (self.Yref - Y)**2)]
      if isinstance(initial_guess, int):
          initial_guess = self.thetaref[initial_guess]
      theta = minimize(
          lambda x: self.__dist_sq(x[0], X, Y),
          [initial_guess],
          jac=True
      ).x[0]
      if eval_ey:
          p = self.__spline(theta)
          dp = self.__spline(theta, 1)
          psi = arctan2(dp[1], dp[0])
          ey = -sin(psi) * (X - p[0]) + cos(psi) * (Y - p[1])
          return theta, ey
      return theta

  def __dist_sq(self, theta, X, Y, eval_gradient=True):
      p = self.__spline(theta)
      d = (p[0] - X)**2 + (p[1] - Y)**2
      if eval_gradient:
          dp = self.__spline(theta, 1)
          return d, 2 * (dp[0] * (p[0] - X) + dp[1] * (p[1] - Y))
      return d

  @property
  def Xref(self):
    return self.disc_coords[:,0]

  @property
  def Yref(self):
    return self.disc_coords[:,1]

  @property
  def thetaref(self):
    return self.sref
  
  @property
  def border_right(self):
    return np.full(self.disc_coords.shape[0], 0.35)

  @property
  def border_left(self):
    return np.full(self.disc_coords.shape[0], 0.35)
  
  def getCenterLineError(self, x, y):
    return getContouringError(x, y, self.disc_coords, self.center_spline)

  def createTrack(self):
    self.coords = getTrajectories(self.row_num, self.col_num)
    self.actions = getActions([0, 0], self.coords)

    self.disc_coords, _, self.sref  = genCenterCoords(self.coords)
    self.kapparef = getCenterCurvature(self.disc_coords)

    self.center_spline = getCenterSpline(self.disc_coords)
    self.__spline = CubicSpline(self.thetaref, self.disc_coords, bc_type="periodic")

  def getFrenetpath(self, state):
    '''
    s: arc length (=thetaref)
    ec: min_distance
    psi: orientation
    Vx: Longitudinal
    Vy: Lateral
    omega: Turning rate
    D: Duty cycle
    delta: Steering angle change rate
    '''
    # target waypoints and curvature
    self.tx = self.Xref
    self.ty = self.Yref
    self.tc = getCenterCurvature(self.disc_coords)
    self.csp = CubicSpline(self.thetaref, self.disc_coords, bc_type="periodic")

    # obstacle lists
    self.ob = np.array([[20.0, 10.0],
                   [30.0, 6.0],
                   [30.0, 8.0],
                   [35.0, 8.0],
                   [50.0, 3.0]
                   ])

    # initial state
    self.c_speed = (state[3]**2 + state[4]**2)**0.5  # current speed [m/s]
    print("speed: ",self.c_speed)
    self.c_accel = 0.0 # current acceleration [m/ss]
    self.c_d_d = state[4]  # current lateral speed [m/s]
    self.c_d_dd = state[6]  # current lateral acceleration [m/s]
    s0, self.c_d = self.get_theta(*state[:2], initial_guess=None, eval_ey=True)  # current course position, current lateral position [m]
    self.s0 = s0**0.5

    path = frenet_optimal_trajectory.frenet_optimal_planning(
        self.csp, self.s0, self.c_speed, self.c_accel, self.c_d, self.c_d_d, self.c_d_dd, self.ob)

    # Update
    state[0] = path.x[1]
    state[1] = path.y[1]
    state[2] = path.yaw[1]
    state[3] = path.s_d[1]
    state[4] = path.d_d[1]

    state[5] = path.d_dd[1] # omega orientation
    state[6] = path.d_dd[1] # delta steering angle
    state[7] = path.d_dd[1] # D Duty-cycle = throttle

    return state  

def dwa(track, state):
  dwa = dynamic_window_approach
  config = dwa.Config()

  # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
  x = np.array([state[0], state[1], state[2], np.hypot(state[3], state[4]), state[5]])

  # goal position [x(m), y(m)]
  config.theta, _ = track.get_theta(*state[:2], initial_guess=config.theta, eval_ey=True)
  # print(config.theta)
  goal = track(config.theta + config.lookahead)
  # print(goal[0], goal[1])

  # input [forward speed, yaw_rate]

  config.robot_type = dwa.RobotType.rectangle
  ob = config.ob

  u, predicted_trajectory = dwa.dwa_control(x, config, goal, ob)
  x = dwa.motion(x, u, config.dt)  # simulate robot

  state[0] = x[0]
  state[1] = x[1]
  state[2] = x[2]
  state[3] = (x[3]*np.cos(x[4]))
  state[4] = (x[3]*np.sin(x[4]))
  state[5] = x[4]

  return state, predicted_trajectory



def getCenterCurvature(disc_coords):
  dx = np.gradient(disc_coords[:,0])
  dy = np.gradient(disc_coords[:,1])
  ddx = np.gradient(dx)
  ddy = np.gradient(dy)
  k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))

  return k
  


def getTrajectories(row_num, col_num):
  while True:
    graph = generateGraph(row_num, col_num)

    start = (0, 0)
    end = (0, 4)

    path = aStar(graph, start, end)
    # print(path)
    if (path is not None):
      path.append((0, 3))
      path.insert(0, (0, 1))
      path.insert(0, (0, 2))

      path.reverse()
      path = np.asarray(path)
      path = path[:, ::-1]

      return path

def generateGraph(row_num, col_num):
  mat = []
  for y in range(row_num):
    row = []
    for x in range(col_num):
      row.append(tileStatus(x, y))
    mat.append(row)

  return mat

def tileStatus(x, y):
  # Start Sections
  if (x==1 and y==0):
    return 1
  if (x==2 and y==0):
    return 1
  if (x==3 and y==0):
    return 1

  # Fixed Wall
  if (x==1 and y==1):
    return 1
  if (x==1 and y==2):
    return 1

  # Random Wall Generation
  if (random.random() < 0.15):
    return 1
  else:
    return 0

def getActions(current, trj):
  actions = []

  for i in range(len(trj)):
    x = trj[i][0]
    y = trj[i][1]

    x_diff = current[0] - x
    y_diff = current[1] - y

    if (x_diff == 1):
      actions.append("W")
    elif (x_diff == -1):
      actions.append("E")
    elif (y_diff == 1):
      actions.append("S")
    elif (y_diff == -1):
      actions.append("N")

    current = [x, y]
  
  return actions

def genCenterCoords(coords):
  resolution = 21
  disc_coords = np.empty((0, 2))
  kapparef = np.empty(0)
  for i, c in enumerate(coords):
    x_diff_prev = coords[i][0] - coords[i-1][0]
    if i+1 == len(coords):
      x_diff_next = coords[0][0] - coords[i][0]
    else:
      x_diff_next = coords[i+1][0] - coords[i][0]

    y_diff_prev = coords[i][1] - coords[i-1][1]
    if i+1 == len(coords):
      y_diff_next = coords[0][1] - coords[i][1]
    else:
      y_diff_next = coords[i+1][1] - coords[i][1]

    map_type = ""
    if (x_diff_prev == x_diff_next == 1):
      map_type = "straight_hor_r"
    elif (y_diff_prev == y_diff_next == 1):
      map_type = "straight_ver_d"
    if (x_diff_prev == x_diff_next == -1):
      map_type = "straight_hor_l"
    elif (y_diff_prev == y_diff_next == -1):
      map_type = "straight_ver_u"
    elif (x_diff_prev == 1 and y_diff_next == 1):
      map_type = "curve_4+"
    elif (x_diff_prev == 1 and y_diff_next == -1):
      map_type = "curve_1-"
    elif (x_diff_prev == -1 and y_diff_next == 1):
      map_type = "curve_3-"
    elif (x_diff_prev == -1 and y_diff_next == -1):
      map_type = "curve_2+"
    elif (y_diff_prev == 1 and x_diff_next == 1):
      map_type = "curve_2-"
    elif (y_diff_prev == 1 and x_diff_next == -1):
      map_type = "curve_1+"
    elif (y_diff_prev == -1 and x_diff_next == 1):
      map_type = "curve_3+"
    elif (y_diff_prev == -1 and x_diff_next == -1):
      map_type = "curve_4-"

    if map_type == "straight_hor_r":
      cs = np.linspace((c[0]-0.5,c[1]), (c[0]+0.5,c[1]), resolution)
      k = np.full(resolution,0)
    if map_type == "straight_hor_l":
      cs = np.linspace((c[0]+0.5,c[1]), (c[0]-0.5,c[1]), resolution)
      k = np.full(resolution,0)
    elif map_type == "straight_ver_d":
      cs = np.linspace((c[0],c[1]-0.5), (c[0],c[1]+0.5), resolution)
      k = np.full(resolution,0)
    elif map_type == "straight_ver_u":
      cs = np.linspace((c[0],c[1]+0.5), (c[0],c[1]-0.5), resolution)
      k = np.full(resolution,0)
    elif map_type == "curve_1+":
      cs = []
      k = np.full(resolution,2)
      center = [c[0]-0.5, c[1]-0.5]
      for i in range(resolution):
        idx = i
        _c = [center[0] + 0.5*np.cos((np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin((np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_1-":
      cs = []
      k = np.full(resolution,-2)
      center = [c[0]-0.5, c[1]-0.5]
      for i in range(resolution):
        idx = resolution - i - 1
        _c = [center[0] + 0.5*np.cos((np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin((np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_2+":
      cs = []
      k = np.full(resolution,2)
      center = [c[0]+0.5, c[1]-0.5]
      for i in range(resolution):
        idx = i
        _c = [center[0] + 0.5*np.cos(np.pi/2+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi/2+(np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_2-":
      cs = []
      k = np.full(resolution,-2)
      center = [c[0]+0.5, c[1]-0.5]
      for i in range(resolution):
        idx = resolution - i - 1
        _c = [center[0] + 0.5*np.cos(np.pi/2+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi/2+(np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_3+":
      cs = []
      k = np.full(resolution,2)
      center = [c[0]+0.5, c[1]+0.5]
      for i in range(resolution):
        idx = i
        _c = [center[0] + 0.5*np.cos(np.pi+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi+(np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_3-":
      cs = []
      k = np.full(resolution,-2)
      center = [c[0]+0.5, c[1]+0.5]
      for i in range(resolution):
        idx = resolution - i - 1
        _c = [center[0] + 0.5*np.cos(np.pi+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi+(np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_4+":
      cs = []
      k = np.full(resolution,2)
      center = [c[0]-0.5, c[1]+0.5]
      for i in range(resolution):
        idx = i
        _c = [center[0] + 0.5*np.cos(np.pi*1.5+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi*1.5+(np.pi/2)*(idx/resolution))]
        cs.append(_c)
    elif map_type == "curve_4-":
      cs = []
      k = np.full(resolution,-2)
      center = [c[0]-0.5, c[1]+0.5]
      for i in range(resolution):
        idx = resolution - i - 1
        _c = [center[0] + 0.5*np.cos(np.pi*1.5+(np.pi/2)*(idx/resolution)), center[1] + 0.5*np.sin(np.pi*1.5+(np.pi/2)*(idx/resolution))]
        cs.append(_c)

    disc_coords = np.concatenate((disc_coords, cs))
    kapparef = np.concatenate((kapparef, k))
  
  sref = []
  for i in range(disc_coords.shape[0]):
    xref = disc_coords[i,0]
    yref = disc_coords[i,1]
    if i == 0:
      s_ = [(xref**2 + yref**2)**0.5]
    else:
      s_ = [sref[-1] + ((xref - disc_coords[i-1,0])**2 + (yref - disc_coords[i-1,1])**2)**0.5]
      if s_ == sref[-1]:
        xref = (disc_coords[i,0] + disc_coords[i+1,0]) / 2
        yref = (disc_coords[i,1] + disc_coords[i+1,1]) / 2
        disc_coords[i,0] = xref
        disc_coords[i,1] = yref
        s_ = [sref[-1] + ((xref - disc_coords[i-1,0])**2 + (yref - disc_coords[i-1,1])**2)**0.5]
    sref = np.concatenate((sref,s_))

  # for i in range(kapparef.shape[0]):
  #   if i < (kapparef.shape[0]-1) and kapparef[i] != kapparef[i+1]:
  #     kapparef[i] = (kapparef[i] + kapparef[i+1]) / 2

  # print("sref\n",sref)
  # print(disc_coords.shape)
  # print(kapparef.shape)
  # print(sref.shape)
  # print(disc_coords[:,0])
  # print(disc_coords[:,1])
  # print(kapparef)
  # print(sref)
  # plt.scatter(disc_coords[:,0], disc_coords[:,1])
  # plt.grid(True)
  # plt.show()

  return disc_coords, kapparef, sref

def getCenterSpline(disc_coords):
  # disc_coords = genCenterCoords(coords)

  thetaref = 2 * np.pi * np.linspace(0, 1, disc_coords.shape[0])
  center_spline = CubicSpline(thetaref, disc_coords, bc_type="periodic")
  xs = 2 * np.pi * np.linspace(0, 1, disc_coords.shape[0])

  return center_spline

def getContouringError(X, Y, disc_coords, center_spline):
  thetaref = 2 * np.pi * np.linspace(0, 1, disc_coords.shape[0])
  ref_idx = np.argmin((disc_coords[:,0] - X)**2 + (disc_coords[:,1] - Y)**2)
  initial_guess = thetaref[ref_idx]

  def dist_sq(spline, theta, X, Y, eval_gradient=True):
    p = spline(theta)
    d = (p[0] - X)**2 + (p[1] - Y)**2
    if eval_gradient:
        dp = spline(theta, 1)
        return d, 2 * (dp[0] * (p[0] - X) + dp[1] * (p[1] - Y))
    return d

  theta = minimize(
      lambda x: dist_sq(center_spline, x[0], X, Y),
      [initial_guess],
      jac=True
  ).x[0]

  p = center_spline(theta)
  dp = center_spline(theta, 1)
  psi = np.arctan2(dp[1], dp[0])
  ey = -np.sin(psi) * (X - p[0]) + np.cos(psi) * (Y - p[1])

  return ey

def getCenterError(X, Y):
  disc_coords = genCenterCoords(coords)
  center_spline = getCenterSpline(disc_coords)

  error = getContouringError(X, Y, disc_coords, center_spline)

  return error


class Node:
  def __init__(self, parent=None, position=None):
    self.parent = parent
    self.position = position

    self.g = 0
    self.h = 0
    self.f = 0

  def __eq__(self, other):
    return self.position == other.position


def heuristic(node, goal, D=1, D2=2 ** 0.5):  # Diagonal Distance
  dx = abs(node.position[0] - goal.position[0])
  dy = abs(node.position[1] - goal.position[1])
  return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def aStar(maze, start, end):
  # startNode와 endNode 초기화
  startNode = Node(None, start)
  endNode = Node(None, end)

  # openList, closedList 초기화
  openList = []
  closedList = []

  # openList에 시작 노드 추가
  openList.append(startNode)

  # endNode를 찾을 때까지 실행
  while openList:

    # 현재 노드 지정
    currentNode = openList[0]
    currentIdx = 0

    # 이미 같은 노드가 openList에 있고, f 값이 더 크면
    # currentNode를 openList안에 있는 값으로 교체
    for index, item in enumerate(openList):
      if item.f < currentNode.f:
        currentNode = item
        currentIdx = index

    # openList에서 제거하고 closedList에 추가
    openList.pop(currentIdx)
    closedList.append(currentNode)

    # 현재 노드가 목적지면 current.position 추가하고
    # current의 부모로 이동
    if currentNode == endNode:
      path = []
      current = currentNode
      while current is not None:
        # maze 길을 표시하려면 주석 해제
        # x, y = current.position
        # maze[x][y] = 7
        path.append(current.position)
        current = current.parent
      return path[::-1]  # reverse

    children = []
    # 인접한 xy좌표 전부
    for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0)]:

      # 노드 위치 업데이트
      nodePosition = (
        currentNode.position[0] + newPosition[0],  # X
        currentNode.position[1] + newPosition[1])  # Y

      # 미로 maze index 범위 안에 있어야함
      within_range_criteria = [
        nodePosition[0] > (len(maze) - 1),
        nodePosition[0] < 0,
        nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
        nodePosition[1] < 0,
      ]

      if any(within_range_criteria):  # 하나라도 true면 범위 밖임
        continue

      # 장애물이 있으면 다른 위치 불러오기
      if maze[nodePosition[0]][nodePosition[1]] != 0:
        continue

      new_node = Node(currentNode, nodePosition)
      children.append(new_node)

    # 자식들 모두 loop
    for child in children:

      # 자식이 closedList에 있으면 continue
      if child in closedList:
        continue

      # f, g, h값 업데이트
      child.g = currentNode.g + 1
      child.h = ((child.position[0] - endNode.position[0]) **
                  2) + ((child.position[1] - endNode.position[1]) ** 2)
      # child.h = heuristic(child, endNode) 다른 휴리스틱
      # print("position:", child.position) 거리 추정 값 보기
      # print("from child to goal:", child.h)

      child.f = child.g + child.h

      # 자식이 openList에 있으고, g값이 더 크면 continue
      if len([openNode for openNode in openList
              if child == openNode and child.g > openNode.g]) > 0:
        continue

      openList.append(child)

if __name__ == "__main__":
  row_num = 5
  col_num = 7

  coords = getTrajectories(row_num, col_num)
  actions = getActions([0, 0], coords)

  disc_coords = genCenterCoords(coords)
  center_spline = getCenterSpline(disc_coords)

  pos = [2, 0.2]
  error = getContouringError(pos[0], pos[1], disc_coords, center_spline)

# print(np.linalg.norm(ey))