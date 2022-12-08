from casadi import sin, cos, tanh, atan, atan2, fabs, sign, fmax, sqrt
from numpy import array, clip

class VehicleModel:

    def __init__(self,
        m=0.041,
        Iz=27.8e-6,
        lf=0.029,
        lr=0.033,
        Cm1=0.287,
        Cm2=0.0545,
        Cr0=0.0518,
        Cr2=0.00035,
        Cr3=5,
        Br=3.3852,
        Cr=1.2691,
        Dr=0.1737,
        Bf=2.579,
        Cf=1.2,
        Df=0.192,
        L=0.24,
        W=0.12,
        delta_min=-0.8,
        delta_max=0.8,
        D_min=-1.0,
        D_max=1.0,
        ddelta_min=-1,
        ddelta_max=1,
        dD_min = -1,
        dD_max = 1
    ):

        self.m = m
        self.Iz = Iz
        self.lf = lf
        self.lr = lr
        self.Cm1 = Cm1
        self.Cm2 = Cm2
        self.Cr0=Cr0
        self.Cr2=Cr2
        self.Cr3=Cr3
        self.Br = Br
        self.Cr = Cr
        self.Dr = Dr
        self.Bf = Bf
        self.Cf = Cf
        self.Df = Df
        self.L = L
        self.W = W

        self.ddelta_min = ddelta_min
        self.ddelta_max = ddelta_max
        self.dD_min = dD_min
        self.dD_max = dD_max

        self.delta_min = delta_min
        self.delta_max = delta_max
        self.D_min = D_min
        self.D_max = D_max


    def get_tire_slip_angle(self, vx, vy, omega, delta):

        vxnew = fmax(vx, 0.1)

        alphaf = -atan2(self.lf * omega + vy, vxnew) + delta
        alphar =  atan2(self.lr * omega - vy, vxnew)

        return alphaf, alphar


    def get_tire_force(self, vx, vy, omega, delta, D):

        alphaf, alphar = self.get_tire_slip_angle(vx, vy, omega, delta)

        Ffy = self.Df * sin(self.Cf * atan(self.Bf * alphaf))
        Fry = self.Dr * sin(self.Cr * atan(self.Br * alphar))

        Frx = (self.Cm1 - self.Cm2 * fabs(vx)) * D - self.Cr0 * tanh(self.Cr3 * vx) - self.Cr2 * vx**2 * sign(vx)

        return Ffy, Fry, Frx


    def f(self, X, Y, psi, vx, vy, omega, delta, D, ddelta, dD):
        '''
        Contunuons-time dynamics model.
        Arguments
        ---------
        X: X-axis position of the Vehicle on the global coordinates.
        Y: Y-axis position of the Vehicle on the global coordinates.
        psi: Heading angle of the Vehicle on the global coordinates.
        vx: Longitudinal velocity of the vehicle.
        vy: Lateral velocity of the vehicle.
        omega: Turning rate of the vehicle.
        delta: Steering angle.
        D: Duty-cycle of the PWM signal of the DC motor.
        ddelta: Steering angle change rate.
        '''
        # [delta, D] = clip([ddelta, dD], [self.delta_min, self.D_min], [self.delta_max, self.D_max])

        Ffy, Fry, Frx = self.get_tire_force(vx, vy, omega, delta, D)

        cos_psi = cos(psi)
        sin_psi = sin(psi)
        cos_delta = cos(delta)

        dx = vx * cos_psi - vy * sin_psi
        dy = vx * sin_psi + vy * cos_psi
        dpsi = omega
        dvx = (Frx - Ffy * sin(delta)) / self.m + vy * omega
        dvy = (Fry + Ffy * cos_delta) / self.m - vx * omega
        domega = (Ffy * self.lf * cos_delta - Fry * self.lr) / self.Iz

        return dx, dy, dpsi, dvx, dvy, domega, ddelta, dD


    def footprint(self, x=0.0, y=0.0, psi=0.0):
        cos_psi = cos(psi)
        sin_psi = sin(psi)
        vertices = array([
            [cos_psi, -sin_psi],
            [sin_psi,  cos_psi]
        ]) @ array([
            [self.L, self.L, -self.L, -self.L],
            [self.W, -self.W, -self.W, self.W],
        ]) / 2 + array([[x], [y]])
        return vertices[0, :], vertices[1, :]


    @property
    def safe_distance(self):
        return ((self.W/2)**2 + (self.L/2)**2)**0.5
