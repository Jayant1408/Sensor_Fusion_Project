# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import numpy as np
# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

# ---------------------------------------------------------------------
# Kalman filter class for prediction and update of tracked object state
# ---------------------------------------------------------------------

class Filter:
    '''Kalman filter class'''
    # Initialize state dimensionality and parameters
    def __init__(self):
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q
    # -----------------------------------------------------------------
    # Define the system matrix F for constant velocity motion
    # -----------------------------------------------------------------
    def F(self):
        ############
        # TODO Step 1: Implement and return system matrix F
        ############
        return np.array([[1.,0.,0.,self.dt, 0.,0.],
                         [0., 1., 0., 0., self.dt, 0.],
                         [0.,0.,1.,0.,0.,self.dt],
                         [0.,0.,0.,1.,0.,0.],
                         [0.,0.,0.,0.,1.,0.],
                         [0.,0.,0.,0.,0.,1.]])

        # return 0
        
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Define the process noise covariance matrix Q
    # -----------------------------------------------------------------
    def Q(self):
        ############
        # TODO Step 2: Implement and return process noise covariance Q
        ############
         ### Compute the system matrix in the case that `dt` has changed
        _F = self.F()
        ### Discretising the continuous model
        # Assuming noise through acceleration is equal in x, y, and z
        _Q = np.diag([0., 0., 0., self.q, self.q, self.q])
        # The matrix exponential
        # Here the integral factor is evaluated from t=0 to t=dt
        _integral_factor = np.array([
            [self.dt / 3, 0., 0., self.dt / 2, 0., 0.],
            [0., self.dt / 3., 0., 0., self.dt / 2, 0.],
            [0., 0., self.dt / 3, 0., 0., self.dt / 2],
            [self.dt / 2, 0., 0., self.dt, 0., 0.],
            [0., self.dt / 2, 0., 0., self.dt, 0.],
            [0., 0., self.dt / 2, 0., 0., self.dt]])
        #Compute process noise covariance
        QT = _integral_factor * np.matmul(_F @ _Q, _F.T)
        return QT.T
    
        # return 0
        
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Predict the next state and covariance using system model
    # -----------------------------------------------------------------
    def predict(self, track):
        ############
        # TODO Step 3: Predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        _F = self.F()
        _Q = self.Q()
        _x = _F @ track.x
        _P = np.matmul(_F @ track.P, _F.T) + _Q
        track.set_x(_x)
        track.set_P(_P)
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Update state and covariance with a new measurement
    # -----------------------------------------------------------------
    def update(self, track, meas):
        ############
        # TODO Step 4: Update state x and covariance P with associated measurement, save x and P in track
        ############
        _gamma = self.gamma(track, meas)
        _H = meas.sensor.get_H(track.x)
        _S = self.S(track, meas, _H)

        _K = np.matmul(track.P @ _H.T, np.linalg.inv(_S))

        _x = track.x + _K @ _gamma

        track.set_x(_x)
        _I = np.identity(n = self.dim_state)
        _P = (_I - np.matmul(_K,_H)) @ track.P

        track.set_P(_P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    # -----------------------------------------------------------------
    # Compute residual (difference between actual and expected measurement)
    # -----------------------------------------------------------------
    def gamma(self, track, meas):
        ############
        # TODO Step 5: calculate and return residual gamma
        ############

        return meas.z - meas.sensor.get_hx(track.x)
            
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Compute innovation covariance S
    # -----------------------------------------------------------------
    def S(self, track, meas, H):
        ############
        # TODO Step 6: calculate and return covariance of residual S
        ############

        return np.matmul(H @ track.P, H.T) + meas.R
        
        ############
        # END student code
        ############ 