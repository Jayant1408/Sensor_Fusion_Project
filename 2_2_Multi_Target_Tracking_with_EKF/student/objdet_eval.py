# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import numpy as np
import matplotlib
# matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well  
matplotlib.use('Agg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Object detection tools and helper functions
import misc.objdet_tools as tools

# -----------------------------------------------------------------------------
# Match detections with ground truth labels and compute IoU + deviations
# -----------------------------------------------------------------------------
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
    # Iterate through ground truth labels and match with predictions
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # Exclude all labels from statistics which are not considered valid
            
            # Compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## Step 1 : Extract the four corners of the current label bounding-box
            box = label.box
            box_1 = tools.compute_box_corners(
                            box.center_x,
                            box.center_y,
                            box.width,
                            box.length,
                            box.heading
            )

            ## Step 2 : Loop over all detected objects
            for det in detections:

                # Step 3: Compute center deviations
                _id, x, y, z, _h, w, l, yaw = det
                box_2 = tools.compute_box_corners(x,y,w,l,yaw)
                ## Step 4 : Compute the center distance between label and detection bounding-box in x, y, and z
                dist_x = np.array(box.center_x - x).item()
                dist_y = np.array(box.center_y - y).item()
                dist_z = np.array(box.center_z - z).item()
                ## Step 5 : Compute the intersection over union (IOU) between label and detection bounding-box
                try:
                    poly_1 = Polygon(box_1)
                    poly_2 = Polygon(box_2)
                    intersection = poly_1.intersection(poly_2).area
                    union = poly_1.union(poly_2).area
                    iou = intersection / union
                except Exception as err:
                    print(f"Encountered '{err}' error in IoU calculation")
                # Step 6: If IoU exceeds threshold, register match
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x,dist_y,dist_z])
                    true_positives += 1
            #######
            ####### ID_S4_EX1 END #######     
            
        # Retain best match (highest IoU) for each label
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # -----------------------------------------------------------------------------
    # Compute precision, recall, and false positives/negatives
    # -----------------------------------------------------------------------------

    ## Step 1 : Compute the total number of positives present in the scene
    # all_positives = 0
    all_positives = labels_valid.sum()

    ## Step 2 : Compute the number of false negatives
    # false_negatives = 0
    false_negatives = all_positives - true_positives

    ## Step 3 : Compute the number of false positives
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance

# -----------------------------------------------------------------------------
# Aggregate and visualize detection performance across all frames
# -----------------------------------------------------------------------------
def compute_performance_stats(det_performance_all, configs_det):

    # Extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')
    # -----------------------------------------------------------------------------
    # Compute global precision and recall
    # -----------------------------------------------------------------------------
    ## Step 1 : Extract the total number of positives, true positives, false negatives and false positives
    all_positive_sum, true_positive_sum, false_negative_sum, false_positive_sum = np.asarray(pos_negs).sum(axis = 0)
    ## Step 2 : Compute precision
    precision = true_positive_sum / float(true_positive_sum + false_positive_sum)    
    ## Step 3 : Compute recall 
    recall = true_positive_sum / float(true_positive_sum + false_negative_sum)

    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # Serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # Compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # Plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
#     plt.show()
    plt.savefig("/home/workspace/Figures/output_plot.png")


