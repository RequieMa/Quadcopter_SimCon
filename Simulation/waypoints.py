# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np

def makeWaypoints():
    v_average = 1.6
    wp_ini = np.array([0, 0, -5])
    wp = np.array([
        wp_ini,
        [2, 2, 1],
        [-2, 3, -3],
        [-2, -1, -3],
        [3, -2, 1],
        wp_ini
    ])
    yaw_ini = 0    
    yaw = np.array([
        yaw_ini, 
        20, 
        -90, 
        120, 
        45, 
        yaw_ini
    ])
    return wp, np.deg2rad(yaw.astype(float)), v_average
