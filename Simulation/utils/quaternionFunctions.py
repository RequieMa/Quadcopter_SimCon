# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy.linalg import norm

# Normalize quaternion, or any vector
def vectNormalize(q):
    return q / norm(q)

# Quaternion multiplication
def quatMultiply(q, p):
    qw, qx, qy, qz = q
    Q = np.array([
        [qw, -qx, -qy, -qz],
        [qx,  qw, -qz,  qy],
        [qy,  qz,  qw, -qx],
        [qz, -qy,  qx,  qw]
    ])
    return Q @ p

# Inverse quaternion
def inverse(q):
    qinv = np.array([q[0], -q[1], -q[2], -q[3]]) / norm(q)
    return qinv