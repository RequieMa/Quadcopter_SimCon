# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from quadFiles.initQuad import sys_params
import utils
import config

class Quadcopter:
    def __init__(self):
        self.params = sys_params()
        
        # Command for initial stable hover
        # m * g / 4 = kTh * omega^2 => omega
        # torque = kTo * omega^2
        # omega = cmd * c1 + c0
        # ---------------------------
        self.m = self.params["m"]
        self.g = self.params["g"]
        self.kTh = self.params["kTh"]
        self.kTo = self.params["kTo"]
        c1 = self.params["motor_c1"]
        c0 = self.params["motor_c0"]
        thr_hover = self.m * self.g / 4.0
        omega_squared = thr_hover / self.kTh
        omega_hover = np.sqrt(omega_squared)

        self.params["FF"] = (omega_hover - c0) / c1 # Feed-Forward Command for Hover
        self.params["omega_hover"] = omega_hover # Motor Speed for Hover
        self.params["thr_hover"] = thr_hover # Motor Thrust for Hover  
        self.thr = np.full(4, thr_hover)
        self.tor = np.full(4, self.kTo * omega_squared)

        # Initial State (Quaternion)
        # ---------------------------
        quat = utils.eulerZYX2quat(0, 0, 0) # psi, theta, phi in rad
        s = [
            0, 0, 0, # s (x, y, z)
            quat[0], quat[1], quat[2], quat[3], # quaternion
            0, 0, 0, # s_dot (dx, dy, dz)
            0, 0, 0, # omega_dot (p, q, r)
            omega_hover, 0, # omega1: Hovering motor speed, motor acc
            omega_hover, 0, # omega2
            omega_hover, 0, # omega3
            omega_hover, 0, # omega4
        ]
        self.state = np.array(s)

        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.motor_omega = np.array([*self.state[13:20:2]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)

        self.extended_state()
        self.forces()
     
    def extended_state(self):
        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.R = utils.quat2R(self.quat)

        # Euler angles of current state
        YPR = utils.quat2EulerZYX(self.quat)
        self.euler = YPR[::-1] # flip YPR so that euler state = phi, theta, psi
        self.psi   = YPR[0]
        self.theta = YPR[1]
        self.phi   = YPR[2]

    def forces(self):
        # Rotor thrusts and torques
        self.thr = self.kTh * self.motor_omega * self.motor_omega
        self.tor = self.kTo * self.motor_omega * self.motor_omega

    def d_state(self, t, state, cmd, wind):
        # Import Params
        # ---------------------------    
        x_arm = self.params["x_arm"]
        y_arm = self.params["y_arm"]
        I_xx, I_yy, I_zz = self.params["I_inertia"].diagonal()
        Cd = self.params["Cd"]
        tau = self.params["tau"]
        kp = self.params["kp"]
        damp = self.params["damp"]
        rotor_omega_min = self.params["rotor_omega_min"]
        rotor_omega_max = self.params["rotor_omega_max"]
        I_motor = self.params["I_motor"]
        uP = 1 if config.usePrecession else 0
    
        # Import State Vector
        # ---------------------------  
        [
            qw, qx, qy, qz,
            dx, dy, dz,
            p, q, r,
            omega1, d_omega1,
            omega2, d_omega2,
            omega3, d_omega3,
            omega4, d_omega4
        ] = state[3:21]
        motor_omega = np.array([omega1, omega2, omega3, omega4])
        d_motor_omega = np.array([d_omega1, d_omega2, d_omega3, d_omega4])

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        u_ctrl = np.array([*cmd[:4]])
        dd_omega = (-2.0 * damp * tau * d_motor_omega - motor_omega + kp * u_ctrl) / tau**2
        motor_omega = np.clip(motor_omega, rotor_omega_min, rotor_omega_max)
        thrust = self.kTh * motor_omega * motor_omega
        torque = self.kTo * motor_omega * motor_omega
    
        thr_m1, thr_m2, thr_m3, thr_m4 = thrust
        tor_m1, tor_m2, tor_m3, tor_m4 = torque

        # Wind Model
        # ---------------------------
        velW, qW1, qW2 = wind.randomWind(t)
        # velW = 0

        # velW = 5          # m/s
        # qW1 = 0*deg2rad    # Wind heading
        # qW2 = 60*deg2rad     # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU)
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        total_thrust = np.sum(thrust)
        diff_omega = omega1 - omega2 + omega3 - omega4
        x_vel_wind = velW * np.cos(qW1) * np.cos(qW2) - dx
        y_vel_wind = velW * np.sin(qW1) * np.cos(qW2) - dy
        z_vel_wind = velW * np.sin(qW2) + dz
        d_dynamics = np.array([
            dx, dy, dz,
            -0.5 * p * qx - 0.5 * q * qy - 0.5 * qz * r,
            0.5 * p * qw - 0.5 * q * qz + 0.5 * qy * r,
            0.5 * p * qz + 0.5 * q * qw - 0.5 * qx * r,
            -0.5 * p * qy + 0.5 * q * qx + 0.5 * qw * r,
            (Cd * x_vel_wind**2 - 2 * (qw*qy + qx*qz) * total_thrust) / self.m,
            (Cd * y_vel_wind**2 + 2 * (qw*qx - qy*qz) * total_thrust) / self.m,
            (-Cd * z_vel_wind**2 - (qw**2 - qx**2 - qy**2 + qz**2) * total_thrust + self.m * self.g) / self.m,
            ((I_yy - I_zz) * q * r - uP * I_motor * diff_omega * q + (thr_m1 - thr_m2 - thr_m3 + thr_m4) * y_arm) / I_xx, # uP activates or deactivates the use of gyroscopic precession.
            ((I_zz - I_xx) * p * r + uP * I_motor * diff_omega * p + (thr_m1 + thr_m2 - thr_m3 - thr_m4) * x_arm) / I_yy, # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
            ((I_xx - I_yy) * p * q - tor_m1 + tor_m2 - tor_m3 + tor_m4) / I_zz
        ])
    
        # State Derivative Vector
        # ---------------------------
        sdot = np.zeros(21)
        sdot[:13] = d_dynamics
        sdot[13:20:2] = d_motor_omega
        sdot[14:21:2] = dd_omega
        return sdot

    def update(self, Ts, quad_state):
        prev_vel = self.vel
        prev_omega = self.omega

        self.pos = quad_state[0:3]
        self.quat = quad_state[3:7]
        self.vel = quad_state[7:10]
        self.omega = quad_state[10:13]
        self.motor_omega = np.array([*quad_state[13:20:2]])

        self.vel_dot = (self.vel - prev_vel) / Ts
        self.omega_dot = (self.omega - prev_omega) / Ts

        self.extended_state()
        self.forces()
