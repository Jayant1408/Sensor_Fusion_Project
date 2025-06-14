# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import numpy as np
from scipy.stats.distributions import chi2

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

# ---------------------------------------------------------------------
# Association class using Mahalanobis distance and nearest neighbor rule
# ---------------------------------------------------------------------
class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    # -----------------------------------------------------------------
    # Construct association matrix between tracks and measurements
    # -----------------------------------------------------------------    
    def associate(self, track_list, meas_list, KF):
             
        ############
        # TODO Step 1: Association:
        # - Replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - Update list of unassigned measurements and unassigned tracks
        ############
        
        # The following only works for at most one track and one measurement
        # self.association_matrix = np.matrix([]) # reset matrix
        # self.unassigned_tracks = [] # reset lists
        # self.unassigned_meas = []
        
        # if len(meas_list) > 0:
        #     self.unassigned_meas = [0]
        # if len(track_list) > 0:
        #     self.unassigned_tracks = [0]
        # if len(meas_list) > 0 and len(track_list) > 0: 
        #     self.association_matrix = np.matrix([[0]])

        self.unassigned_tracks = list(range(len(track_list)))
        self.unassigned_meas = list(range(len(meas_list)))

        self.association_matrix = np.array(np.full((len(track_list), len(meas_list)), fill_value = np.inf))

        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                dist = self.MHD(track, meas, KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i, j] = dist
        ############
        # END student code
        ############ 
                
    # -----------------------------------------------------------------
    # Retrieve closest track-measurement pair from association matrix
    # -----------------------------------------------------------------
    def get_closest_track_and_meas(self):
        ############
        # TODO Step 2: Find closest track and measurement:
        # - Find minimum entry in association matrix
        # - Delete row and column
        # - Remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - Return this track and measurement
        ############

        # The following only works for at most one track and one measurement
        # update_track = 0
        # update_meas = 0
        
        # Remove from list
        # self.unassigned_tracks.remove(update_track) 
        # self.unassigned_meas.remove(update_meas)
        # self.association_matrix = np.matrix([])
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan
        else:
            pass

        idx_track, idx_meas = np.unravel_index(
            indices = np.argmin(self.association_matrix, axis = None),
            shape = self.association_matrix.shape
        )

        update_track = self.unassigned_tracks[idx_track]
        update_meas = self.unassigned_meas[idx_meas]

        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.delete(
            arr = self.association_matrix,
            obj = idx_track,
            axis = 0
        )
        self.association_matrix = np.delete(
            arr = self.association_matrix,
            obj = idx_meas,
            axis = 1
        )   
        ############
        # END student code
        ############ 
        return update_track, update_meas     
    # -----------------------------------------------------------------
    # Check if Mahalanobis distance is within the gating threshold
    # -----------------------------------------------------------------
    def gating(self, MHD, sensor): 
        ############
        # TODO Step 3: Return True if measurement lies inside gate, otherwise False
        ############
        
        ppf = chi2.ppf(q = params.gating_threshold, df = sensor.dim_meas)
        if MHD < ppf:
            return True
        else:
            return False
           
        
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Compute Mahalanobis distance for a track-measurement pair
    # -----------------------------------------------------------------        
    def MHD(self, track, meas, KF):
        ############
        # TODO Step 4: Calculate and return Mahalanobis distance
        ############

        H = meas.sensor.get_H(track.x)
        gamma = meas.z - meas.sensor.get_hx(track.x)
        S = KF.S(track,meas,H)

        dist = np.matmul(gamma.T @ np.linalg.inv(S), gamma)

        return dist
        
        ############
        # END student code
        ############ 
    # -----------------------------------------------------------------
    # Full association and update pipeline using Kalman Filter
    # -----------------------------------------------------------------    
    def associate_and_update(self, manager, meas_list, KF):
        # Associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # Update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # Search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # Check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # Update score and track state 
            manager.handle_updated_track(track)
            
            # Save updated track
            manager.track_list[ind_track] = track
            
        # Run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)