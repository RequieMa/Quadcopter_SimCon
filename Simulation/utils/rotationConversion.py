# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

class Quat:
    def __init__(self):
        self.w = 0
        self.v = np.zeros(3)

    @property
    def eulerZYX(self):
        qw = self.w
        qx, qy, qz = self.v
        psi = np.arctan2(2.0 * (qw * qz + qx * qy), 1 - 2.0 * (qy * qy + qz * qz))
        theta = np.arcsin(2.0 * (qw * qy - qx * qz))
        phi = np.arctan2(2.0 * (qw * qx + qy * qz), 1 - 2.0 * (qx * qx + qy * qy))
        return np.array([phi, theta, psi])
    
    @property
    def R(self):
        qw = self.w
        qx, qy, qz = self.v
        rot_mat = np.array([
            [1 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qw*qz), 2.0 * (qw*qy + qx*qz)],
            [2.0 * (qx*qy + qw*qz) , 1 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qw*qx)],
            [2.0 * (qx*qz - qw*qy), 2.0 * (qw*qx + qy*qz), 1 - 2.0 * (qx*qx + qy*qy)],
        ])
        return rot_mat
    
    @property
    def q(self):
        return np.hstack((self.w, self.v))

    @q.setter
    def q(self, value):
        self.w = value[0]
        self.v = value[1:]
    
    @property
    def inverse(self):
        return self.conjugate().normalize()

    def from_eulerZYX(self, phi, theta, psi):
        c_psi = np.cos(0.5 * psi)
        c_theta = np.cos(0.5 * theta)
        c_phi = np.cos(0.5 * phi)
        s_psi = np.sin(0.5 * psi)
        s_theta = np.sin(0.5 * theta)
        s_phi = np.sin(0.5 * phi)

        self.w = c_psi*c_theta*c_phi + s_psi*s_theta*s_phi
        self.v = np.array([
            c_psi*c_theta*s_phi - s_psi*s_theta*c_phi,
            c_psi*s_theta*c_phi + s_psi*c_theta*s_phi,
            s_psi*c_theta*c_phi - c_psi*s_theta*s_phi
        ])
        self.normalize()
        return self

    def from_R(self, R):
        [
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33],
        ] = R

        # From page 68 of MotionGenesis book
        tr = R11 + R22 + R33
        if tr > R11 and tr > R22 and tr > R33:
            qw = 0.5 * np.sqrt(1 + tr)
            r = 0.25 / qw
            qx = (R32 - R23) * r
            qy = (R13 - R31) * r
            qz = (R21 - R12) * r
        elif R11 > R22 and R11 > R33:
            qx = 0.5 * np.sqrt(1 - tr + 2 * R11)
            r = 0.25 / qx
            qw = (R32 - R23) * r
            qy = (R12 + R21) * r
            qz = (R13 + R31) * r
        elif R22 > R33:
            qy = 0.5 * np.sqrt(1 - tr + 2 * R22)
            r = 0.25 / qy
            qw = (R13 - R31) * r
            qx = (R12 + R21) * r
            qz = (R23 + R32) * r
        else:
            qz = 0.5 * np.sqrt(1 - tr + 2 * R33)
            r = 0.25 / qz
            qw = (R21 - R12) * r
            qx = (R13 + R31) * r
            qy = (R23 + R32) * r

        self.w = qw
        self.v = np.sign(qw) * np.array([qx, qy, qz])
        self.normalize()
        return self

    def normalize(self):
        self.q /= norm(self.q)
        return self

    def conjugate(self):
        result = Quat()
        result.w = self.w
        result.v = -self.v
        return result

    def __mul__(self, quat):
        result = Quat()
        result.w = self.w * quat.w - self.v @ quat.v
        result.v = self.w * quat.v + quat.w * self.v + np.cross(self.v, quat.v)
        return result

def eulerZYX2quat(psi, theta, phi):
    # For ZYX, Yaw-Pitch-Roll
    c_psi = np.cos(0.5 * psi)
    c_theta = np.cos(0.5 * theta)
    c_phi = np.cos(0.5 * phi)
    s_psi = np.sin(0.5 * psi)
    s_theta = np.sin(0.5 * theta)
    s_phi = np.sin(0.5 * phi)

    # qw,qx,qy,qz
    q = np.array([
        c_psi*c_theta*c_phi + s_psi*s_theta*s_phi,
        c_psi*c_theta*s_phi - s_psi*s_theta*c_phi,
        c_psi*s_theta*c_phi + s_psi*c_theta*s_phi,
        s_psi*c_theta*c_phi - c_psi*s_theta*s_phi
    ])

    q /= norm(q)
    return q

def quat2R(q):
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qw*qz), 2.0 * (qw*qy + qx*qz)],
        [2.0 * (qx*qy + qw*qz) , 1 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qw*qx)],
        [2.0 * (qx*qz - qw*qy), 2.0 * (qw*qx + qy*qz), 1 - 2.0 * (qx*qx + qy*qy)],
    ])
    return R

def quat2EulerZYX(q):
    # YPR = [Yaw, pitch, roll] = [psi, theta, phi]
    qw, qx, qy, qz = q
    psi = np.arctan2(2.0 * (qw*qz + qx*qy), 1 - 2.0 * (qy*qy + qz*qz))
    theta = np.arcsin(2.0 * (qw*qy - qx*qz))
    phi = np.arctan2(2.0 * (qw*qx + qy*qz), 1 - 2.0 * (qx*qx + qy*qy))
    return np.array([psi, theta, phi])

def RotToQuat(R):
    [
        [R11, R12, R13],
        [R21, R22, R23],
        [R31, R32, R33],
    ] = R

    # From page 68 of MotionGenesis book
    tr = R11 + R22 + R33
    if tr > R11 and tr > R22 and tr > R33:
        qw = 0.5 * np.sqrt(1 + tr)
        r = 0.25 / qw
        qx = (R32 - R23) * r
        qy = (R13 - R31) * r
        qz = (R21 - R12) * r
    elif R11 > R22 and R11 > R33:
        qx = 0.5 * np.sqrt(1 - tr + 2 * R11)
        r = 0.25 / qx
        qw = (R32 - R23) * r
        qy = (R12 + R21) * r
        qz = (R13 + R31) * r
    elif R22 > R33:
        qy = 0.5 * np.sqrt(1 - tr + 2 * R22)
        r = 0.25 / qy
        qw = (R13 - R31) * r
        qx = (R12 + R21) * r
        qz = (R23 + R32) * r
    else:
        qz = 0.5 * np.sqrt(1 - tr + 2 * R33)
        r = 0.25 / qz
        qw = (R21 - R12) * r
        qx = (R13 + R31) * r
        qy = (R23 + R32) * r

    q = np.array([qw, qx, qy, qz])
    q *= np.sign(qw)
    q /= norm(q)
    return q


# def RPYtoRot_ZYX(RPY):
    
#     phi = RPY[0]
#     theta = RPY[1]
#     psi = RPY[2]
    
# #    R = np.array([[np.cos(psi)*np.cos(theta) - np.sin(phi)*np.sin(psi)*np.sin(theta),
# #                   np.cos(theta)*np.sin(psi) + np.cos(psi)*np.sin(phi)*np.sin(theta), 
# #                   -np.cos(phi)*np.sin(theta)],
# #                  [-np.cos(phi)*np.sin(psi),
# #                   np.cos(phi)*np.cos(psi),
# #                   np.sin(phi)],
# #                  [np.cos(psi)*np.sin(theta) + np.cos(theta)*np.sin(phi)*np.sin(psi),
# #                   np.sin(psi)*np.sin(theta) - np.cos(psi)*np.cos(theta)*np.sin(phi),
# #                   np.cos(phi)*np.cos(theta)]])
    
#     psi = psi
#     theta = theta
#     phi = phi
#     # Rotation ZYX from page 277 of MotionGenesis book
#     R = np.array([[np.cos(psi)*np.cos(theta),
#                    -np.sin(psi)*np.cos(phi) + np.sin(theta)*np.sin(phi)*np.cos(psi), 
#                    np.sin(psi)*np.sin(phi) + np.sin(theta)*np.cos(psi)*np.cos(phi)],
#                   [np.sin(psi)*np.cos(theta),
#                    np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi),
#                    -np.sin(phi)*np.cos(psi) + np.sin(psi)*np.sin(theta)*np.cos(phi)],
#                   [-np.sin(theta),
#                    np.sin(phi)*np.cos(theta),
#                    np.cos(theta)*np.cos(phi)]])
    
#     return R

