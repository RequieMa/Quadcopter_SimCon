# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy.linalg import norm
import utils

class Control:
    # Set PID Gains and Max Values
    # ---------------------------
    # Position P gains
    Px = Py = 1.0
    Pz = 1.0
    pos_P_gain = np.array([Px, Py])

    # Velocity P-D gains
    Pxdot = Pydot = 5.0
    Dxdot = Dydot = 0.5
    Ixdot = Iydot = 5.0

    Pzdot = 4.0
    Dzdot = 0.5
    Izdot = 5.0

    vel_P_gain = np.array([Pxdot, Pydot])
    vel_D_gain = np.array([Dxdot, Dydot])
    vel_I_gain = np.array([Ixdot, Iydot])

    # Attitude P gains
    Ptheta = Pphi = 8.0
    Ppsi = 1.5
    PpsiStrong = 8
    att_P_gain = np.array([Pphi, Ptheta, Ppsi])

    # Max Velocities
    saturateVel_separetely = False
    uMax = 5.0
    vMax = 5.0
    wMax = 5.0
    velMax = np.array([uMax, vMax, wMax])
    velMaxAll = 5.0

    # Max tilt
    tiltMax = np.deg2rad(50.0)

    # Rate P-D gains
    Pq = Pp = 1.5
    Dq = Dp = 0.04

    Pr = 1.0
    Dr = 0.1

    rate_P_gain = np.array([Pp, Pq, Pr])
    rate_D_gain = np.array([Dp, Dq, Dr])

    # Max Rate
    pMax = np.deg2rad(200.0)
    qMax = np.deg2rad(200.0)
    rMax = np.deg2rad(150.0)
    rateMax = np.array([pMax, qMax, rMax])

    def __init__(self, quad, traj):
        self.quad = quad
        self.traj = traj
        self.sDesCalc = np.zeros(16)
        self.motor_cmd = np.ones(4) * self.quad.params["omega_hover"]
        self.thr_int = np.zeros(3)
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5 * (self.att_P_gain[0] + self.att_P_gain[1])
        self.yaw_w = np.clip(self.att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)
        self.att_P_gain[2] = roll_pitch_gain

        self.pos_sp = np.zeros(3)
        self.vel_sp = np.zeros(3)
        self.acc_sp = np.zeros(3)
        self.thrust_sp = np.zeros(3)
        self.eul_sp = np.zeros(3)
        self.pqr_sp = np.zeros(3)
        self.yawFF = np.zeros(3)

    def controller(self, Ts):
        sDes = self.traj.sDes
        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp = sDes[0:3]
        self.vel_sp = sDes[3:6]
        self.acc_sp = sDes[6:9]
        self.thrust_sp = sDes[9:12]
        self.eul_sp = sDes[12:15]
        self.pqr_sp = sDes[15:18]
        self.yawFF = sDes[18]
        
        # Select Controller
        # ---------------------------
        self.z_pos_control()
        self.xy_pos_control()
        self.saturateVel()
        self.z_vel_control(Ts)
        self.xy_vel_control(Ts)
        self.thrustToAttitude()
        self.attitude_control()
        self.rate_control()

        # Mixer
        # --------------------------- 
        self.motor_cmd = utils.mixerFM(self.quad, norm(self.thrust_sp), self.rateCtrl)
        
        # Add calculated Desired States
        # ---------------------------         
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp

    def z_pos_control(self):
        # Z Position Control
        # --------------------------- 
        pos_z_error = self.pos_sp[2] - self.quad.pos[2]
        self.vel_sp[2] += self.Pz * pos_z_error
        
    def xy_pos_control(self):
        # XY Position Control
        # --------------------------- 
        pos_xy_error = self.pos_sp[0:2] - self.quad.pos[0:2]
        self.vel_sp[0:2] += self.pos_P_gain * pos_xy_error
        
    def saturateVel(self):
        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (prefered)
        if self.saturateVel_separetely:
            self.vel_sp = np.clip(self.vel_sp, -self.velMax, self.velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if totalVel_sp > self.velMaxAll:
                self.vel_sp *= self.velMaxAll / totalVel_sp

    def z_vel_control(self, Ts):
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_z_error = self.vel_sp[2] - self.quad.vel[2]
        thrust_z_sp = self.Pzdot * vel_z_error - self.Dzdot * self.quad.vel_dot[2] + self.quad.params["m"] * (self.acc_sp[2] - self.quad.params["g"]) + self.thr_int[2]
        
        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -self.quad.params["minThr"]
        uMin = -self.quad.params["maxThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not stop_int_D:
            self.thr_int[2] += self.Izdot * vel_z_error * Ts * self.quad.params["has_intergral_ctrl"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), self.quad.params["maxThr"]) * np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)
    
    def xy_vel_control(self, Ts):
        # XYZ Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = self.vel_sp[0:2] - self.quad.vel[0:2]
        thrust_xy_sp = self.vel_P_gain * vel_xy_error - self.vel_D_gain * self.quad.vel_dot[0:2] + self.quad.params["m"] * self.acc_sp[0:2] + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2]) * np.tan(self.tiltMax)
        thrust_max_xy = np.sqrt(self.quad.params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if self.thrust_sp[0:2].T @ self.thrust_sp[0:2] > thrust_max_xy**2:
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp / mag * thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        arw_gain = 2.0 / self.vel_P_gain
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += self.vel_I_gain * vel_err_lim * Ts * self.quad.params["has_intergral_ctrl"]
    
    def thrustToAttitude(self):
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]

        # Desired body_z axis direction
        body_z = -utils.vectNormalize(self.thrust_sp)
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([
            -np.sin(yaw_sp), 
            np.cos(yaw_sp), 
            0.0
        ])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        body_x = utils.vectNormalize(body_x)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([
            body_x, 
            body_y, 
            body_z
        ]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = utils.RotToQuat(R_sp)
        
    def attitude_control(self):
        # Current thrust orientation e_z and desired thrust orientation e_z_d
        e_z = self.quad.R[:, 2]
        e_z_d = -utils.vectNormalize(self.thrust_sp)

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = e_z @ e_z_d + norm(e_z) * norm(e_z_d)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = utils.vectNormalize(qe_red)
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        self.qd_red = utils.quatMultiply(qe_red, self.quad.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = utils.quatMultiply(utils.inverse(self.qd_red), self.qd_full)
        q_mix *= np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        self.qd = utils.quatMultiply(self.qd_red, np.array([
            np.cos(self.yaw_w * np.arccos(q_mix[0])), 
            0, 
            0, 
            np.sin(self.yaw_w * np.arcsin(q_mix[3]))
        ]))

        # Resulting error quaternion
        self.qe = utils.quatMultiply(utils.inverse(self.quad.quat), self.qd)

        # Create rate setpoint from quaternion error
        self.rate_sp = 2.0 * np.sign(self.qe[0]) * self.qe[1:] * self.att_P_gain
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -self.rMax, self.rMax)

        # Add Yaw rate feed-forward
        self.rate_sp += utils.quat2R(utils.inverse(self.quad.quat))[:, 2] * self.yawFF
        self.rate_sp = np.clip(self.rate_sp, -self.rateMax, self.rateMax)

    def rate_control(self):
        rate_error = self.rate_sp - self.quad.omega
        self.rateCtrl = self.rate_P_gain * rate_error - self.rate_D_gain * self.quad.omega_dot     # Be sure it is right sign for the D part
        



    