# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# General package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# Add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Model-related imports
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode
from tools.objdet_models.resnet.utils.evaluation_utils import post_processing
from tools.objdet_models.resnet.utils.torch_utils import _sigmoid

# -----------------------------------------------------------------------------
# Load model-specific configuration parameters
# -----------------------------------------------------------------------------
def load_configs_model(model_name='darknet', configs=None):

    # Init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # Get path to the parent directory
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # Define model-specific parameters
    
    # Darknet model (YOLOv4-based)
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        # ResNet with Feature Pyramid Network (FPN) backbone
        print("student task ID_S3_EX1-3")
        # Path to the pre-trained model weights
        configs.model_path = os.path.join(
            parent_path, 'tools', 'objdet_models', 'resnet'
        )
        configs.pretrained_filename = os.path.join(
            configs.model_path,'pretrained', 'fpn_resnet_18_epoch_300.pth'
        )
        ### Step 2 : Adding and updating configs defined in `sfa/test.py`
        # The name of the model architecture
        configs.arch = 'fpn_resnet'
        # The folder name to use for saving logs, trained models, outputs, etc.
        configs.saved_fn = 'fpn-resnet'    # Equivalent to 'fpn_resnet_18'
        # The subfolder to save current model outputs in '../results/{saved_fn}'
        configs.rel_results_folder = 'results_sequence_1_resnet'
        # The path to the pre-trained model
        configs.pretrained_path = configs.pretrained_filename
        # Number of convolutional layers to use
        configs.num_layers = 18
        # The number of top 'K'
        configs.K = 50
        # If True, cuda is not used
        configs.no_cuda = False
        # GPU index to use
        configs.gpu_idx = 0
        # Subset of dataset to run and debug
        configs.num_samples = None
        # Number of threads for loading data 
        configs.num_workers = 1
        # Number of samples per mini-batch
        configs.batch_size = 1
        # The non-maximum suppression (NMS) score to use
        configs.nms_thresh = 0.4
        # The peak threshold
        configs.peak_thresh = 0.2
        # The minimum confidence threshold to use for detections
        configs.conf_thresh = 0.5
        # The minimum Intersection over Union (IoU) threshold to use
        configs.min_iou = 0.5
        # If True, output image of testing phase will be saved
        configs.save_test_output = False
        # Type of test output (can be one of: ['image', 'video'])
        configs.output_format = 'image'
        # The video filename to use (if the output format is 'video')
        configs.output_video_fn = 'out_fpn_resnet'
        # The width of the output
        configs.output_width = 608
        configs.pin_memory = True
        # Is False when testing on a single GPU only
        configs.distributed = False
        # The input Bird's-Eye View (BEV) image size
        configs.input_size = (608, 608)
        # The bounding box anchor size for balanced L1 loss
        configs.hm_size = (152, 152)
        # The down-sampling ratio S s.t. (H/S, W/S, C)
        # for C num. classes and heatmap for main center of size (H, W)
        configs.down_ratio = 4
        # Max. number of predictions to keep, i.e., the maximum number
        # of predictions whose center confidences are greater than 0.2
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        # The num. channels in the convolutional layer for the output head
        # 64 for ResNet and 256 for DLA
        configs.head_conv = 64
        configs.num_classes = 3
        # The center offset, i.e., (H/S, W/S, 2)
        configs.num_center_offset = 2
        # The z-coordinate dimension, i.e., (H/S, W/S, 1)
        configs.num_z = 1
        # The number of channels in the input, i.e., (H, W, 3)
        configs.num_dim = 3
        # Model estimates the complex-valued yaw angles
        # The `im` and `re` fractions are directly regressed using `l1_loss`
        configs.num_direction = 2    # sin, cos
        # Parameters used for the regression tasks, i.e., the
        # Sigmoid / Focal / L1 / Balanced L1 loss functions
        configs.heads = {
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4
        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")
    
    # Path to the dataset folder relative to the root folder
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(
        configs.root_dir, 'dataset'
    )
    # Path to the subfolder in 'results' for the current run of this model
    if configs.save_test_output:
        configs.result_dir = os.path.join(
            configs.root_dir, 'results', 
            configs.saved_fn, configs.rel_results_folder
    )
    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs

# -----------------------------------------------------------------------------
# Load global configurations (including BEV, colors, etc.)
# -----------------------------------------------------------------------------
def load_configs(model_name='fpn_resnet', configs=None):

    # Init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # Bird's-Eye View parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # Load model-specific parameters
    configs = load_configs_model(model_name, configs)

    # Visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs

# -----------------------------------------------------------------------------
# Instantiate the model and load weights
# -----------------------------------------------------------------------------
def create_model(configs):

    # Instantiate the model based on architecture
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # Create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        model = fpn_resnet.get_pose_net(
            num_layers=configs.num_layers,
            heads = configs.heads,
            head_conv = configs.head_conv,
            imagenet_pretrained=configs.imagenet_pretrained
        )
        print("student task ID_S3_EX1-4")

        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # Load weights and set evaluation model 
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  
    model.eval()          

    return model


# -----------------------------------------------------------------------------
# Perform inference and post-processing on input BEV maps
# -----------------------------------------------------------------------------
def detect_objects(input_bev_maps, model, configs):

    # Deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # Perform inference
        outputs = model(input_bev_maps)

        # Decode model output into target object format
        if 'darknet' in configs.arch:

            # Post-process YOLOv4 output
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # Decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(
                hm_cen=outputs['hm_cen'],
                cen_offset=outputs['cen_offset'],
                direction = outputs['direction'],
                z_coor=outputs['z_coor'],
                dim = outputs['dim'],
                K = configs.K
            )

            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections=detections, configs=configs)

            detections = detections[0][1]  # Only get detections for 'Car' class (class id = 1)
            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    # -----------------------------------------------------------------------------
    # Convert network output to 3D bounding boxes in real-world coordinates
    # -----------------------------------------------------------------------------
    print("student task ID_S3_EX2")
    objects = [] 
    ## step 1 : check whether there are any detections
    # if not detections:
    #     return objects
    if detections is None or (isinstance(detections, np.ndarray) and detections.size == 0):
        return objects

    for obj in detections:
        _id, _x, _y, _z, _h, _w, _l, _yaw = obj
        ### Step 3 : Perform the coordinate conversion
        # Here we use the limits for x, y and z set in the configs structure
        x = _y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        y = _x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        y -= (configs.lim_y[1] - configs.lim_y[0]) / 2
        w = _w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        l = _l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        z = _z
        yaw = _yaw
        h = _h
        ### Step 4 : Append the current object to the `objects` array
        if ((x >= configs.lim_x[0] and x <= configs.lim_x[1]) and
            (y >= configs.lim_y[0] and y <= configs.lim_y[1]) and 
            (z >= configs.lim_z[0] and z <= configs.lim_z[1])
        ):
            # Making sure the bounding box is within the limits of the image
            objects.append([1, x, y, z, h, w, l, yaw])
        
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    

