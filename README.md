# Orientation Tracking with Projected Gradient Descent
This project has 3 parts for computations described below and one part for plotting, the code for which is in `plotting.py`

## Abstract ([PDF](https://drive.google.com/uc?export=download&id=1SmmFjOQdmE1U0cPkOFgOtPyPE5qW-9Ys))
This project investigates three-dimensional orientation tracking of a rotating rigid body using Inertial Measurement
Unit (IMU) data, with application to panoramic image construction from onboard camera images. Angular velocity and linear
acceleration measurements from a gyroscope and accelerometer
are used to estimate orientation in the absence of continuous
ground-truth sensing. Raw IMU data are first calibrated to
remove sensor biases and converted from analog-to-digital units
into physically meaningful quantities. An initial orientation
trajectory is obtained through discrete-time quaternion integration of gyroscope measurements. Orientation estimation is then
formulated as a constrained trajectory optimization problem over
unit quaternions that combines gyroscope motion dynamics and
accelerometer gravity observations. The optimization is solved
using projected gradient descent to enforce unit-norm quaternion
constraints. Estimated orientations are evaluated against VICON
motion capture ground truth on training datasets, demonstrat
ing significant improvement over direct gyroscope integration,
particularly for roll and pitch. Yaw accuracy remains limited
due to the lack of complete information about rotation about the
gravity axis. The optimized orientation estimates are then used
to align camera images and generate panoramic reconstructions
via equirectangular projection. Results demonstrate that this
approach (projected gradient descent on trajectory quaternions)
enables accurate orientation tracking and effective panorama
generation using IMU and camera data alone.

## IMU Utilities (`imu_utils.py`)
1) **IMU Calibration**: Calibrate IMU measurements by removing sensor biases and converting raw ADC values to physical units (acceleration in g, angular velocity in rad/s.) using scale factors derived from sensor specifications in IMU reference sheet

2) **Gyroscope Integration**: Perform discrete-time integration of calibrated gyroscope data to obtain orientation trajectory quaternions using the motion model $f(\mathbf{q_t}, τ_t*\mathbf{ω_t})$, starting from initial quaternion $\mathbf{q_0}$ = [1, 0, 0, 0]

3) **Euler Angle Conversion**: Convert quaternion trajectories to Euler angles (roll, pitch, yaw) for visualization and comparison with VICON ground truth

4) **VICON Data Loading**: Load and convert VICON rotation matrices to Euler angles for ground truth comparison

## Optimization (`optimization.py`)
1) **PyTorch Quaternion Operations**: Implement quaternion multiplication, inverse, exponential map, and logarithm map with automatic differentiation support for gradient computation.

2) **Motion Model**: Predict next orientation using gyroscope measurements

3) **Observation Model**: Predict accelerometer measurements by rotating gravity vector to body frame

4) **Cost Function**: Implement cost function combining motion model error (gyroscope consistency)and observation model error (accelerometer-gravity alignment).

5) **Projected Gradient Descent**: Optimize quaternion trajectory with projection onto unit quaternion manifold (normalization after each gradient step).

## Panorama Display (`panorama.py`)
1) **Camera Data Loading**: Load RGB images (H=240, W=320) and timestamps from camera data files.

2) **Temporal Synchronization**: Match each camera image timestamp to the closest-in-the-past IMU quaternion estimate (t_imu ≤ t_cam).

3) **Quaternion to Rotation Matrix**: Convert orientation quaternions q=[w,x,y,z] to 3×3 rotation matrices for coordinate transformation.

4) **Pixel to Camera Ray**: Project each camera pixel (u,v) to a 3D unit ray in camera frame using pinhole camera model with FOV (60° horizontal, 45° vertical).

5) **Camera to World Transformation**: Rotate rays from camera frame to world frame using rotation matrices: ray_world = R × ray_camera.

6) **Equirectangular Projection**: Convert world-frame rays (x,y,z) to panorama  pixel coordinates by computing:
       
    - Longitude: $\lambda$ = arctan2(y, x)
    - Latitude: $\phi$ = arcsin(z)
    - Map to panorama: u_pano = ($\lambda$+π)/(2π)×width, v_pano = (π/2-$\phi$)/π×height


7) **Image Stitching**: Populate panorama canvas (1000×500 pixels) by assigning RGB values from camera pixels to computed panorama coordinates, using overwrite strategy for overlapping regions.

## Running the Code

### 1) Generate Simple Integration Results (Pre-Optimization)

Calibrates IMU data and performs simple gyroscope integration to verify calibration.

**Training set** (with VICON comparison):
```bash
cd code
python -m imu_utils --set train
```
Outputs: 
- `results/train/simple_integration_comparison_{1-9}.png` (Euler angles vs VICON)
- `results/train/q_simple_{1-9}.npy` (quaternion trajectories)

**Test set** (without VICON):
```bash
cd code
python -m imu_utils --set test
```
Outputs:
- `results/test/simple_integration_results_{10-11}.png` (Euler angles only)
- `results/test/q_simple_{10-11}.npy` (quaternion trajectories)

---

### 2) Run Projected Gradient Descent Optimization

Optimizes orientation estimates using both gyroscope and accelerometer data.

**Training set** (with VICON comparison):
```bash
cd code
python -m optimization --set train
```
Outputs:
- `results/train/optimized_comparison_{1-9}.png` (Optimized vs VICON)
- `results/train/q_optimized_{1-9}.npy` (optimized quaternions)

**Test set** (without VICON):
```bash
cd code
python -m optimization --set test
```
Outputs:
- `results/test/optimized_comparison_{10-11}.png` (Optimized only, or before/after)
- `results/test/q_optimized_{10-11}.npy` (optimized quaternions)

---

### 3) Generate Panoramic Images

Creates panoramas using optimized orientation estimates.

**Training set**:
```bash
cd code
python -m panorama --set train
```
Outputs:
- `results/train/panorama_optimized_{1,2,8,9}.png` (panoramas for datasets with camera data)

**Test set**:
```bash
cd code
python -m panorama --set test
```
Outputs:
- `results/test/panorama_optimized_{10,11}.png` (test set panoramas)

---

### Notes:
- Steps must be run in order (1, 2, 3) as each step depends on outputs from previous steps
- Only datasets 1, 2, 8, 9 (train) and 10, 11 (test) have camera data for panorama generation

- All outputs are saved to `results/train/` or `results/test/` directories

