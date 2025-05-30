# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?


### 4. Can you think of ways to improve your tracking results in the future?

# Project 2.2: Multi-Target Tracking with Extended Kalman Filter
## Final Report


## Introduction

<img src="Final_project_figures\Tracking_results.avi" width="90%" height="90%" alt="Figure 2. Results of the multi-target tracking (MTT) programme evaluated on a sequence obtained from the Waymo Open Dataset.">

In this project, we implemented a multi-target tracking system capable of detecting and tracking vehicles over time using the Waymo Open Dataset. The tracking pipeline used a Single / Sub-optimal Nearest Neighbor (SNN) algorithm with validation gating to improve both runtime and accuracy. The system relied on coordinate transformations between LiDAR and camera sensor frames, particularly for RGB data, which required handling a non-linear measurement model.

To evaluate the results of our multi-target tracking system, we extracted real-world 3D LiDAR detections and RGB images from the `Waymo Open Dataset`. We fused 3D LiDAR detections with camera-based 2D bounding boxes from our `3D_Object_Detection_With_LiDAR_Data` module and assessed tracking performance in real-world driving scenes. Our results demonstrated that the system can track multiple objects reliably and highlight the impact of different components in the pipeline.



## Extended Kalman Filter (EKF)

The `Extended Kalman Filter`(EKF) was used to predict and update an object's state over time based on incoming sensor measurements. It handled non-linear measurement models—such as those from RGB cameras—by linearizing them using the Jacobian of the measurement function $h(\mathbf{x})$, expanded via a multivariate `Taylor series` about the current state estimate.

We implemented the predict and update steps in `filter.py`, which were executed at every time step when a new measurement became available. This iterative process improved estimates of an object’s position and velocity.



## Multi-Target Tracking (MTT)
In order to successfully track multiple objects over time, the system includes `data association` and `track management`. In the `data association` task, incoming measurement are assigned to a new track instance. A track refers to a variable which encapsulates a state estimate and its covariances, together with a set of tracking attributes that describe the quality of the track prediction.



### Data Association

In the data association module, incoming measurements from LiDAR or camera were matched with existing tracks using the `Mahalanobis distance` — a metric that accounts for uncertainty in both the predicted track and the measurement:

 <!-- $\mathrm{z} = \left[p_{x}, p_{y} p_{z} \ldots \right]^{\top}$ and distribution $D$.  -->

$$
D^2 = (\mathbf{z} - \mathbf{h}(\mathbf{x}))^\top S^{-1} (\mathbf{z} - \mathbf{h}(\mathbf{x}))
$$

This helped improve association accuracy over simpler methods like Euclidean distance. We also applied validation gating to exclude improbable associations, enhancing robustness and efficiency.

### Track Management
The track management module was responsible for initializing, confirming, and deleting tracks. New tracks were initialized from LiDAR detections, and a scoring system based on recent detection history determined whether a track should be maintained or removed.

The track score was computed as:

$$
\text{track score} = \frac{\text{detections in last } n \text{ frames}}{n}
$$

This score reflected the consistency of a track being associated with valid detections. A high score indicated a reliable track, while a low score suggested a false positive or a vehicle that had exited the sensor’s field of view. Tracks with low confidence or high uncertainty (based on the covariance matrix $P$) were pruned.

## Sensor Fusion
For this project, we performed `mid-level sensor fusion` by associating 2D camera detections with 3D LiDAR points.  With each sensor having a defined sensor coordinate frame, we were able to translate the coordinates from the LiDAR sensor along the $x$-axis and rotate them about the $z$-axis of the ego-vehicle world frame using calibration parameters, enabling association in 2D space. LiDAR was used to initialize tracks due to its depth accuracy, while both sensors were used for updates.

## Results

We validated our tracking pipeline using both visual outputs and quantitative evaluation via Root Mean Square Error (RMSE) plots. Each step—EKF, track management, data association, and sensor fusion—was evaluated individually to track improvements in performance and runtime.

Execution time was measured using time `python loop_over_dataset.py`, although this included GUI delays from Matplotlib and video export, so actual runtime performance is likely bette

### Extended Kalman Filter
#### Challenges

Initial RMSE scores were high due to a systematic offset in LiDAR measurements—particularly along the $y$-axis—which violated the zero-mean noise assumption of the Kalman Filter.

We addressed this by tuning key parameters for sensor fusion like `delete_threshold` and `max_P`, set to `0.6` and `$3^2$` respectively in `params.py`.



#### Results
##### Single-target tracking

<img src="Final_project_figures/my_tracking_results.avi" width="90%" height="90%" alt="Figure 1. Single-target tracking evaluated on frames [150, 200] in Sequence 2 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 1. Single-target tracking evaluated on frames [150, 200] in Sequence 2 from the Waymo Open Dataset.}
\end{align}
$$

<img src="Final_project_figures\RMSE_single_target.png" width="90%" height="90%" alt="Figure 2. RMSE score of the single-target tracking results evaluated on frames [150, 200] in Sequence 2 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 2. RMSE score of the single-target tracking results evaluated on frames [150, 200] in Sequence 2 from the Waymo Open Dataset.}
\end{align}
$$

Single-target tracking was successful with consistent performance in frames [150, 200] with a field of view set to `configs_det.lim_y = [-5, 10]` using only LiDAR-based EKF prediction and update. 


### Multi-Target Tracking
#### Challenges

In dense environments, tracking became difficult when multiple objects produced nearby measurements. We adopted Mahalanobis distance for better robustness, however SNN had limitations in overlapping gating regions. As a starting point, we suggest exploring the the `Global Nearest Neighbor`(GNN) algorithm in order to find a globally-consistent collection of hypotheses such that an overall score of correct association is maximised.Additionally, the current method scales poorly, as it computes $N \times M$ associations for every frame with $N$ measurements and $M$ tracks.

#### Improvements

To reduce false positives, we introduced a track scoring system and implemented `validation gating` using the residual $\gamma$ and the covariance matrix $S$. Assuming a confidence level of $1 - \alpha$, true measurements outside the gate could still be accounted for probabilistically.
 

#### Results

##### Track management

<img src="out/report/2022-11-23-Output-3-Step-2-Single-Target-Tracking-Results-Sequence-2-Frames-65-100.gif" width="90%" height="90%" alt="Figure 3. Track management (initialisation and deletion) evaluated on frames [65, 100] in Sequence 2 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 3. Track management (initialisation and deletion) evaluated on frames [65, 100] in Sequence 2 from the Waymo Open Dataset.}
\end{align}
$$

With a LiDAR field of view set to `configs_det.lim_y = [-15, 5]`, track initialization and deletion worked reliably and `correctly initialized and deleted tracks`. Confirmed tracks were retained, and false positives were eliminated.

<img src="out/report/2022-11-23-Output-4-Step-2-Single-Target-Tracking-Performance-Evaluation-RMSE.png" width="90%" height="90%" alt="Figure 4. RMSE score of the track management (initialisation and deletion) module for a single target evaluated on frames [65, 100] in Sequence 2 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 4. RMSE score of the track management (initialisation and deletion) module for a single target evaluated on frames [65, 100] in Sequence 2 from the Waymo Open Dataset."}
\end{align}
$$

The single target is tracked successfully as indicated in the RMSE plot as a single line with a fairly consistent RMSE score recorded for the duration of the tracking session. There are no losses due to false negatives shown above.

##### Data association

<img src="out/report/2022-11-23-Output-5-Step-3-Multi-Target-Tracking-Results-Sequence-1-Frames-0-200.gif" width="90%" height="90%" alt="Figure 5. Results of the multi-target tracking (MTT) programme with data association evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 5. Results of the multi-target tracking (MTT) programme with data association evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.}
\end{align}
$$

The above results indicate that our tracking implementation system successfully tracked multiple objects across the entire 200-frame sequence with LiDAR (`configs_det.lim_y = [-25, 25]`). 

While the ungated SNN approach performed reasonably well, adding validation gating led to a notable runtime improvement.


<img src="out/report/2022-11-23-Output-6-Step-3-Multi-Target-Tracking-Performance-Evaluation-RMSE.png" width="90%" height="90%" alt="Figure 6. RMSE score of the multi-target tracking (MTT) programme with data association evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 6. RMSE score of the multi-target tracking (MTT) programme with data association evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.}
\end{align}
$$

RMSE trends were stable and consistent across tracks, showing clear improvements over single-target tracking results


### Sensor Fusion
#### Challenges
Challenges associated with designing of multi-modal sensor fusion algorithms while fusing measurements from two different measurement models are as follows:

Fusing 6D LiDAR measurements ( $\mathrm{z} = \left[p_{x} p_{y} p_{z} v_{x} v_{y} v_{z} \right]^{\top}$ => position and velocity in the three-dimensional vehicle frame) with 2D camera detections ( bounding box coordinate predictions in two-dimensional image space) required precise coordinate transformations and careful calibration. LiDAR provided accurate depth, while the camera lacked depth information but offered rich appearance cues. 

Transforming data between sensor frames involved aligning the LiDAR coordinate frame with the ego-vehicle’s forward $x$-axis using azimuth correction. Camera resectioning projected 3D vehicle-frame points into the image plane using a camera matrix. LiDAR points were converted into homogeneous coordinates for projection.

#### Results

##### Multi-Target Tracking with Camera-LiDAR Sensor Fusion 

<img src="out/report/2022-11-23-Output-7-Step-4-Multi-Target-Sensor-Fusion-Tracking-Results-Sequence-1-Frames-0-200.gif" width="90%" height="90%" alt="Figure 7. Results of the final multi-target tracking (MTT) programme with camera-LiDAR sensor fusion evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 7. Results of the multi-target tracking (MTT) programme with camera-LiDAR sensor fusion evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.}
\end{align}
$$

The MTT system consistently tracks nearby vehicles across 200 frames. We used the full field of view of the LiDAR (`configs.lim_y = [-25, 25]`). Although false-positive (FP) tracks do appear, they are typically recycled quickly across 200 frames.

<img src="out/report/2022-11-23-Output-8-Step-4-Multi-Target-Tracking-Sensor-Fusion-Performance-Evaluation-RMSE.png" width="90%" height="90%" alt="Figure 8. RMSE score of the final multi-target tracking (MTT) programme with camera-LiDAR sensor fusion evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.">

$$
\begin{align}
\textrm{Figure 8. RMSE score of the final multi-target tracking (MTT) programme with camera-LiDAR sensor fusion evaluated on frames [0, 200] in Sequence 1 from the Waymo Open Dataset.}
\end{align}
$$

Above are the RMSE scores for the multi-track setting in Sequence 1 over the 200 frames. 

In our first test (without validation gating), fusion led to a `9.08%` increase in `RMSE` over LiDAR-only tracking. Adding gating improved performance slightly, with a final RMSE of `0.1566`, only `2.17% worse` than the LiDAR-only setup.

Execution time increased from `7m22s (with gating)` to `8m01s (without gating)`—although timing was affected by `plt.pause()` calls, relative differences were consistent.

Comparing the videos from LiDAR-only and fused tracking, both appear to handle nearby vehicle tracks similarly well. Both examples seem to quickly identify and stabilise the nearest vehicle tracks, and both seem to quickly recycle ghost tracks, suggesting that sensor fusion did not provide a clearly visible benefit over one sensor tracking


### In Conclusion


Based on our results, the EKF-based tracking system successfully handled multiple targets over time and effectively managed false positive measurements. 

However, we did not observe significant improvements in tracking accuracy or track management with the addition of camera-LiDAR sensor fusion. Instead, fusion introduced additional runtime overhead without delivering clear benefits in the tested scenarios.

Our evaluation was limited to three sequences from the Waymo Open Dataset, all recorded in daylight under normal traffic conditions with minimal occlusion. Therefore, we cannot generalize our conclusions to more challenging environments such as nighttime, fog, or heavy rain.

Whether camera data can complement LiDAR weaknesses in such scenarios—e.g., false positive returns from road dividers or traffic barriers—remains an open research question.

Given that this project was limited to `mid-level sensor fusion` of camera detections and LiDAR measurements, in future work I would like to explore low-level fusion of raw sensor data via deep learning along with Multi-angle fusion, using additional LiDAR and camera views. In addition to that, I would like to try more accurate motion models like the Kinematic Bicycle Model.


## Closing Remarks

### Alternatives

- Apply `3D reconstruction` using photogrammetry (e.g., Structure from Motion) to estimate depth from monocular camera input and ego-motion.
- Implement a more robust data association algorithm such as `Probabilistic Data Association` (PDA) or `Joint Probabilistic Data Association` (JPDA) filtering.

### Extension of Task

- Tune `max_P` (maximum measurement covariance) to improve deletion of uncertain tracks.
- Replace the constant velocity model with a nonlinear motion model such as the Kinematic Bicycle Model, which considers steering angle $\phi$.
- Improve object dimension estimates (width, length, height) rather than relying solely on unfiltered LiDAR detections in the EKF.


## Future Work

- ⬜️ Fine-tune tracking hyperparameters (e.g., initial state covariance $\sigma$) to reduce RMSE.
- ⬜️ Explore more robust data association methods, such as GNN or PDA filters.
- ⬜️ Compare sensor fusion results against 2D–3D ground truth correspondences in the Waymo Open Dataset.
- ⬜️ Use tracking output for downstream tasks like path planning and motion prediction.

* [2] Konstantinova, P. et al. A Study of a Target Tracking Algorithm Using Global Nearest Neighbor Approach. CompSysTech '03: Proceedings of the 4th International Conference on Computer Systems and Technologies. Association of Computing Machinery. pp.290-295. 2003. [doi:10.1145/973620.973668](https://doi.org/10.1145/973620.973668).

Helpful resources:
* [`2022-11-11-Course-2-Sensor-Fusion-Exercises-Part-2.ipynb` by J. L. Moran | GitHub](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/2-Sensor-Fusion/Exercises/2022-11-11-Course-2-Sensor-Fusion-Exercises-Part-2.ipynb);
* [`2022-11-17-Course-2-Sensor-Fusion-Exercises-Part-3.ipynb` by J. L. Moran | GitHub](https://github.com/jonathanloganmoran/ND0013-Self-Driving-Car-Engineer/blob/main/2-Sensor-Fusion/Exercises/2022-11-17-Course-2-Sensor-Fusion-Exercises-Part-3.ipynb);
* [`07-Kalman-Filter-Math.ipynb` by R. Labbe | GitHub](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb);
* [Introduction to Data Association by B. Collins | Lecture Slides](https://www.cse.psu.edu/~rtc12/CSE598C/datassocPart1.pdf).