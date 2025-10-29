# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy.linalg import inv

def sys_params():
    params = {
        'm': 1.2, # mass (kg)
        'g': 9.81, # gravity (m/s^2)
        'x_arm': 0.16, # arm length (m)
        'y_arm': 0.16, # arm length (m)
        'motor_h': 0.05, # motor height (m)
        'I_inertia': np.array([
            [0.0123, 0,      0     ],
            [0,      0.0123, 0     ],
            [0,      0,      0.0224]
        ]), # Inertial tensor (kg*m^2)
        'I_motor': 2.7e-5, # Rotor moment of inertia (kg*m^2)
        'has_intergral_ctrl': False, # Include integral gains in linear velocity control
        'Cd': 0.1,
        'kTh': 1.076e-5, # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        'kTo': 1.632e-7, # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)

        'minThr': 0.1 * 4, # Minimum total thrust
        'maxThr': 9.18 * 4, # Maximum total thrust
        'rotor_omega_min': 75, # Minimum motor rotation speed (rad/s)
        'rotor_omega_max': 925, # Maximum motor rotation speed (rad/s)

        'tau': 0.015, # Value for second order system for Motor dynamics
        'kp': 1.0, # Value for second order system for Motor dynamics
        'damp': 1.0, # Value for second order system for Motor dynamics

        # omega (rad/s) = cmd * c1 + c0 (cmd in %) -> Use to calculate control
        'motor_c1': 8.49,
        'motor_c0': 74.7
    }
    # params["invI"] = inv(I_inertia) # No use
    # params["interpYaw"] = bool(False)       # Interpolate Yaw setpoints in waypoint trajectory

    params["mixerFMinv"] = inv(makeMixerFM(params)) 
    # params["motordeadband"] = 1   # No use
    # params["ifexpo"] = bool(False)
    # if params["ifexpo"]:
    #     params["maxCmd"] = 100      # cmd (%) min and max
    #     params["minCmd"] = 0.01
    # else:
    #     params["maxCmd"] = 100
    #     params["minCmd"] = 1
    return params

def makeMixerFM(params):
    # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
    x_arm = params["x_arm"]
    y_arm = params["y_arm"]
    kTh = params["kTh"]
    kTo = params["kTo"] 

    # Motor 1 is front left, then clockwise numbering.
    # A mixer like this one allows to find the exact RPM of each motor 
    # given a desired thrust and desired moments.
    # Inspiration for this mixer (or coefficient matrix) and how it is used : 
    # https://link.springer.com/article/10.1007/s13369-017-2433-2 (https://sci-hub.tw/10.1007/s13369-017-2433-2)
    mixerFM = np.array([
        [    kTh,      kTh,      kTh,      kTh],
        [y_arm * kTh, -y_arm * kTh,  -y_arm * kTh, y_arm * kTh],
        [x_arm * kTh,  x_arm * kTh, -x_arm * kTh, -x_arm * kTh],
        [   -kTo,      kTo,     -kTo,      kTo]
    ])
    return mixerFM