# # ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import cv2
import numpy as np
import open3d as o3d
import sys
import torch
import zlib

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# Object detection tools and helper functions
import misc.objdet_tools as tools

# -----------------------------------------------------------------------------
# Visualize raw LiDAR point cloud using Open3D
# -----------------------------------------------------------------------------

def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    def close_window(vis):
        # Notify the window to be closed
        vis.close()
        # Return boolean indicating if `update_geometry` needs to be run
        return False
    print("student task ID_S1_EX2")

    # Step 1 : Initialize open3d visualizer 
    visualiser = o3d.visualization.VisualizerWithKeyCallback()
    str_window = 'Visualising the Waymo Open Dataset: LiDAR Point Cloud data'
    visualiser.create_window(window_name = str_window, width = 1280, height = 720, left = 50 ,top = 50, visible = True)
    visualiser.register_key_callback(key = 262, callback_func = close_window)
    # Step 2 : Create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    # Step 3 : Set points in pcd instance by converting the point-cloud into 3d vectors 
    pcd.points = o3d.utility.Vector3dVector(pcl[:,:3])
    # Step 4 : Add geometry and run visualizer
    visualiser.add_geometry(pcd)
    visualiser.run()

    #######
    ####### ID_S1_EX2 END #######     
       
# -----------------------------------------------------------------------------
# Extract and visualize the range image from Waymo LiDAR data
# -----------------------------------------------------------------------------
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # Step 1: Extract lidar data and range image from the specified LiDAR
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    if len(lidar.ri_return1.range_image_compressed) > 0:
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array (ri.data).reshape(ri.shape.dims)
    # Step 2: Extract range and intensity channels
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]
    # Step 3: Normalize and map to 8-bit range
    MIN_RANGE = 0
    MAX_RANGE = 75 if lidar_name is dataset_pb2.LaserName.TOP else 20    
    # Step 4 : Map the range channel onto an 8-bit scale
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    ri_range = ri_range.astype(np.uint8)
    # Step 5 : Map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    ri_min = np.percentile(ri_intensity,1)
    ri_max = np.percentile(ri_intensity,99)
    np.clip(ri_intensity, a_min = ri_min, a_max = ri_max)
    ri_intensity = np.int_((ri_intensity - ri_min) * 255. / (ri_max - ri_min))
    ri_intensity = ri_intensity.astype(np.uint8)
    # Step 4: Stack both channels vertically
    img_range_intensity = np.vstack((ri_range, ri_intensity))
    img_range_intensity = img_range_intensity.astype(np.uint8)
    # img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity

# -----------------------------------------------------------------------------
# Convert raw LiDAR point cloud to Bird’s Eye View (BEV) tensor
# -----------------------------------------------------------------------------
def bev_from_pcl(lidar_pcl, configs):

    # Step 1: Filter LiDAR points based on predefined detection area
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # Shift ground level to zero
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # Step 2: Convert sensor coordinates to BEV pixel coordinates
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")
    ### Step 1 :  Compute BEV map discretisation
    bev_interval = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    bev_offset = (configs.bev_width + 1) / 2
    ### Step 2 : Transform all matrix x-coordinates into BEV coordinates
    lidar_pcl_cpy = lidar_pcl.copy()
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_interval))
    ### Step 3 : Perform the same operation as in Step 2 for the y-coordinates
    lidar_pcl_cpy[:, 1] = np.int_(
        np.floor(lidar_pcl_cpy[:, 1] / bev_interval) + bev_offset
    )
    # Here we make sure that no negative BEV coordinates occur
    lidar_pcl_cpy[lidar_pcl_cpy < 0.0] = 0.0
    ### Step 4 : Visualise point cloud using `show_pcl` from task ID_S1_EX2
    # if vis:
    show_pcl(lidar_pcl_cpy)
    ####### ID_S2_EX1 END #######     
    ####### ID_S2_EX2 START #######
    ### Summary: Compute intensity layer of the BEV map
    print("student task ID_S2_EX2")
    ### Step 0: Pre-processing
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0
    ### Step 1 : Create a Numpy array filled with zeros
    intensity_map = np.zeros(shape=(configs.bev_height + 1, configs.bev_height + 1))
    ### Step 2 : Re-arrange elements in `lidar_pcl_cpy`
    idxs_intensity = np.lexsort(
        keys=(-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])
    )
    lidar_pcl_top = lidar_pcl_cpy[idxs_intensity]
    ### Step 3 : Extract all points with identical x and y,
    #            s.t. only top-most z-coordinate is kept
    _, idxs_top_unique, counts = np.unique(
            lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    lidar_pcl_top = lidar_pcl_cpy[idxs_top_unique]
    ### Step 4 : Assign the intensity map values
    intensity_vals = lidar_pcl_top[:, 3]
    scale_factor_intensity = np.amax(intensity_vals) - np.amin(intensity_vals)
    intensity_map[np.int_(lidar_pcl_top[:, 0]),
                  np.int_(lidar_pcl_top[:, 1])
                 ] = lidar_pcl_top[:, 3] / scale_factor_intensity
    ### Step 5 : Temporarily visualise the intensity map using OpenCV
    # if vis:
    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
    str_title = "Bird's-eye view (BEV) map: normalised intensity channel values"
    cv2.imwrite(os.path.join("outputs", "img_intensity.png"), img_intensity)
        # cv2.imshow(str_title, img_intensity)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    ####### ID_S2_EX2 END ####### 
    ####### ID_S2_EX3 START #######
    # Summary: Compute height layer of the BEV map
    print("student task ID_S2_EX3")
    ### Step 1 : Create the BEV map array
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    ### Step 2 : Assign the height map values from `lidar_top_pcl`
    scale_factor_height = float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    height_map[np.int_(lidar_pcl_top[:, 0]),
               np.int_(lidar_pcl_top[:, 1])
              ] = lidar_pcl_top[:, 2] / scale_factor_height
    ### Step 3 : Temporarily visualize the height map using OpenCV
    # if vis:
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    str_title = "Bird's-eye view (BEV) map: normalised height channel values"
    cv2.imwrite(os.path.join("outputs", "img_height.png"), img_intensity)
        # cv2.imshow(str_title, img_height)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    ####### ID_S2_EX3 END #######
    ### Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(
            lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True
    )
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]),
                np.int_(lidar_pcl_top[:, 1])
                ] = normalizedCounts
    ### Create a 3-channel BEV map from the individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]
    ### Expand dimension of `bev_map` before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map
    ### Create the tensor from the Bird's-Eye View (BEV) map
    bev_maps = torch.from_numpy(bev_maps)
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


