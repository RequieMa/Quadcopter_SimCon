# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm

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

