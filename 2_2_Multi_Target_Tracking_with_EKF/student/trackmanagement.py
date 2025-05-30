# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import numpy as np
import collections

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

# ---------------------------------------------------------------------
# Track class: manages individual tracked object state and attributes
# ---------------------------------------------------------------------
class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # TODO Step 1: Initialization:
        # - Replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - Initialize track state and track score with appropriate values
        ############

        # self.x = np.matrix([[49.53980697],
        #                 [ 3.41006279],
        #                 [ 0.91790581],
        #                 [ 0.        ],
        #                 [ 0.        ],
        #                 [ 0.        ]])
        # self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])
        self.x = np.ones((6,1))
        sens_z = meas.z
        sens_z = np.vstack((sens_z, np.newaxis))
        sens_z[3] = 1
        sens2veh_T = meas.sensor.sens_to_veh
        self.x[0:4] = sens2veh_T @ sens_z
        self.P = np.zeros((6,6))
        M_rot = meas.sensor.sens_to_veh[0:3,0:3]
        sens_R = meas.R
        self.P[0:3,0:3] = np.matmul(M_rot @ sens_R, M_rot.T)
        self.P[3:6, 3:6] = np.diag([params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])
        self.state = 'initialized'
        self.score = 1. / params.window
        
        ############
        # END student code
        ############ 
               
        # Other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # Use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
            
# ---------------------------------------------------------------------
# Trackmanagement class: manages track lifecycle and logic
# ---------------------------------------------------------------------
class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
    # -----------------------------------------------------------------
    # Handle logic for maintaining active tracks
    # -----------------------------------------------------------------     
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: Implement track management:
        # - Decrease the track score for unassigned tracks
        # - Delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
        
        # Decrease score for unassigned tracks
        _delete_tracks = []
        for i in unassigned_tracks:
            track = self.track_list[i]
            # Check visibility    
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    track.score = track.score - 1. / params.window
                    track.score = max(track.score, 0.)
                    if track.state == 'confirmed':
                        threshold = params.delete_threshold
                    elif (track.state in {'initialized', 'initialised', 'tentative'}):
                        threshold = params.delete_init_threshold
                    else:
                        raise ValueError(f"Invalid track state '{track.state}'")
                    if (track.score < threshold or track.P[0,0] > params.max_P or track.P[1,1] > params.max_P):
                        _delete_tracks.append(track)
                    else:
                        pass
                else:
                    pass
            else:
                pass
            for tracks in _delete_tracks:
                self.delete_track(tracks)

        # delete old tracks   

        ############
        # END student code
        ############ 
            
        # Initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
    # -----------------------------------------------------------------
    # Add a track to the internal track list
    # -----------------------------------------------------------------            
    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
    # -----------------------------------------------------------------
    # Handle updated tracks after successful data association
    # -----------------------------------------------------------------        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 3: Implement track management for updated tracks:
        # - Increase track score
        # - Set track state to 'tentative' or 'confirmed'
        ############

        _new_score = track.score + 1. / params.window
        track.score = min(_new_score,1.0)
        if track.state in {'initialized', 'initialized'}:
            track.state = 'tentative'
        elif track.state == 'tentative':
            if track.score > params.confirmed_threshold:
                track.state = 'confirmed'
            else:
                pass        
        elif track.state == 'confirmed':
            pass
        else:
            raise ValueError(f"Invalid track state '{track.state}'")

        ############
        # END student code
        ############ 