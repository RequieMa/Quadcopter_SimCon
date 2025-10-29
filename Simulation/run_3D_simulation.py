# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time
import cProfile
from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils

class Simulation:
    def __init__(self, Ti, Ts, quad, traj, ctrl, wind, numTimeStep):
        self.Ts = Ts
        self.quad = quad
        self.ctrl = ctrl
        self.wind = wind
        self.traj = traj

        self.integrator = ode(self.quad.d_state).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.quad.state, Ti)

        self.t_all = np.zeros(numTimeStep)
        self.pos_all = np.zeros([numTimeStep, len(quad.pos)])
        self.vel_all = np.zeros([numTimeStep, len(quad.vel)])
        self.quat_all = np.zeros([numTimeStep, len(quad.quat)])
        self.omega_all = np.zeros([numTimeStep, len(quad.omega)])
        self.euler_all = np.zeros([numTimeStep, len(quad.euler)])
        self.sDes_traj_all = np.zeros([numTimeStep, len(traj.sDes)])
        self.sDes_calc_all = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
        self.w_cmd_all = np.zeros([numTimeStep, len(ctrl.motor_cmd)])
        self.wMotor_all = np.zeros([numTimeStep, len(quad.motor_omega)])
        self.thr_all = np.zeros([numTimeStep, len(quad.thr)])
        self.tor_all = np.zeros([numTimeStep, len(quad.tor)])
        
    def run_sim(self, t):
        # Dynamics (using last timestep's commands)
        self.integrator.set_f_params(self.ctrl.motor_cmd, self.wind)
        quad_state = self.integrator.integrate(t, t + self.Ts)
        self.quad.update(self.Ts, quad_state)
        self.traj.get_desired_state(t, self.Ts) # Trajectory for Desired States 
        self.ctrl.controller(self.Ts) # Generate Commands (for next iteration)
        t += self.Ts
        return t
    
    def save_result(self, i, t):
        self.t_all[i] = t
        self.pos_all[i, :] = self.quad.pos
        self.vel_all[i, :] = self.quad.vel
        self.quat_all[i, :] = self.quad.quat
        self.omega_all[i, :] = self.quad.omega
        self.euler_all[i, :] = self.quad.euler
        self.sDes_traj_all[i, :] = self.traj.sDes
        self.sDes_calc_all[i, :] = self.ctrl.sDesCalc
        self.w_cmd_all[i, :] = self.ctrl.motor_cmd
        self.wMotor_all[i, :] = self.quad.motor_omega
        self.thr_all[i, :] = self.quad.thr
        self.tor_all[i, :] = self.quad.tor
    
    def obtain_result(self):
        return [
            self.t_all, 
            self.pos_all, 
            self.vel_all, 
            self.quat_all, 
            self.omega_all, 
            self.euler_all, 
            self.w_cmd_all, 
            self.wMotor_all, 
            self.thr_all, 
            self.tor_all, 
            self.sDes_traj_all, 
            self.sDes_calc_all
        ]
    

def main():
    start_time = time.time()

    # Simulation Setup
    # --------------------------- 
    # Yaw Trajectory Type: follow
    Ti = 0
    Ts = 0.005
    Tf = 20
    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect = 8

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    quad = Quadcopter()
    traj = Trajectory(quad, trajSelect)
    ctrl = Control(quad, traj)
    wind = Wind('SINE', 2.0, 90, -15)
    sim = Simulation(Ti, Ts, quad, traj, ctrl, wind, int(Tf / Ts + 1))

    traj.get_desired_state(Ti, Ts)  
    ctrl.controller(Ts)
    sim.save_result(0, Ti)

    # Simulation
    # ---------------------------
    t = Ti
    step = 1
    while round(t, 3) < Tf:
        t = sim.run_sim(t)
        sim.save_result(step, t)
        step += 1
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))
    [
        t_all, 
        pos_all, vel_all, 
        quat_all, omega_all, euler_all, 
        w_cmd_all, wMotor_all, 
        thr_all, tor_all, 
        sDes_traj_all, sDes_calc_all
    ] = sim.obtain_result()
    # utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, sDes_traj_all, sDes_calc_all)
    utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, traj.xyzType)
    plt.show()

if __name__ == "__main__":
    main()
    # cProfile.run('main()')