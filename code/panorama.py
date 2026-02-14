import numpy as np
import matplotlib.pyplot as plt
from load_data import read_data
from imu_utils import calibrate_imu
from PIL import Image
import logging
from pathlib import Path
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_camera_data(cam_file) -> np.ndarray:
    """
    Load camera images and timestamps from file.
    Parameters:
        cam_file: Path to camera data file
    Returns:
        images: array of images
    """
    cam_data = read_data(cam_file)
    images = cam_data['cam']
    cam_ts = cam_data['ts'].flatten()
    return images, cam_ts

def match_quaternions_to_camera(q_traj, imu_ts, cam_ts) -> np.ndarray:
    """
    Match quaternions to camera timestamps by finding the closest IMU timestamp.
    Parameters:
        q_traj: quaternion trajectory
        imu_ts: IMU timestamps
        cam_ts: camera timestamps 
    Returns:
        q_cam: quaternions matched to camera timestamps
    """
    K = len(cam_ts)
    q_cam = np.zeros((4, K))
    
    for k in range(K):
        idx = np.where(imu_ts <= cam_ts[k])[0]
        
        if len(idx) > 0:
            closest_idx = idx[-1] 
        else:
            closest_idx = 0
        
        q_cam[:, k] = q_traj[:, closest_idx]
    
    return q_cam

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    Parameters:
        q: quaternion
    Returns: 
        result: 3x3 rotation matrix
    """
    w, x, y, z = q
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                   [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x], 
                   [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]])
    return R

def pixel_to_ray(u, v, H, W, fov_h=60, fov_v=45) -> np.ndarray:
    """
    Convert pixel coordinates to 3D ray in camera frame.
    Parameters:
        u: pixel x-coordinate
        v: pixel y-coordinate
        H: image height
        W: image width
        fov_h: horizontal field of view in degrees
        fov_v: vertical field of view in degrees
    Returns:
        ray: 3D unit vector in camera frame
    """
    fov_h_rad = np.radians(fov_h)
    fov_v_rad = np.radians(fov_v)
    fx = W / (2 * np.tan(fov_h_rad / 2)) 
    fy = H / (2 * np.tan(fov_v_rad / 2))  
    cx = W / 2  
    cy = H / 2 
    x = 1.0 
    y = -(u - cx) / fx 
    z = -(v - cy) / fy 
    ray = np.array([x, y, z])
    ray = ray / np.linalg.norm(ray)
    return ray

def ray_to_panorama_coords(ray, pano_width, pano_height) -> tuple:
    """
    Convert 3D ray to panorama pixel coordinates.
    Parameters:
        ray: 3D unit vector
        pano_width: panorama width
        pano_height: panorama height
    Returns:
        pano_u: panorama pixel x-coordinate
        pano_v: panorama pixel y-coordinate
    """
    x, y, z = ray
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))
    pano_u = (lon + np.pi) / (2 * np.pi) * pano_width
    pano_v = (np.pi/2 - lat) / np.pi * pano_height
    return int(pano_u), int(pano_v)

def create_panorama(images, cam_ts, q_traj, imu_ts, pano_width=1000, pano_height=500, fov_h=60, fov_v=45) -> np.ndarray: 
    """
    Create panorama from camera images and orientation quaternions.
    Parameters:
        images: array of camera images
        cam_ts: camera timestamps
        q_traj: orientation quaternions
        imu_ts: IMU timestamps
        pano_width: panorama width
        pano_height: panorama height
        fov_h: camera horizontal FOV in degrees
        fov_v: camera vertical FOV in degrees
    Returns:
        panorama: generated panorama image
    """
    H, W, _, K = images.shape
    
    logging.info(f"Creating panorama from {K} images")
    logging.info(f"Panorama size: {pano_width} x {pano_height}")
    logging.info(f"Camera FOV(degrees): {fov_h} (horizontal) x {fov_v} (vertical)")
    panorama = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    filled = np.zeros((pano_height, pano_width), dtype=bool)
    q_cam = match_quaternions_to_camera(q_traj, imu_ts, cam_ts)
    
    for k in range(K):
        if k % 10 == 0:
            logging.info(f"Processing image {k+1}/{K}")

        R = quaternion_to_rotation_matrix(q_cam[:, k])
        img = images[..., k]

        for v in range(0, H, 2): 
            for u in range(0, W, 2):
                ray_camera = pixel_to_ray(u, v, H, W, fov_h, fov_v)
                ray_world = R @ ray_camera
                pano_u, pano_v = ray_to_panorama_coords(ray_world, pano_width, pano_height)
                
                if 0 <= pano_u < pano_width and 0 <= pano_v < pano_height:
                    color = img[v, u, :]
                    panorama[pano_v, pano_u, :] = color
                    filled[pano_v, pano_u] = True
    
    logging.info(f"Panorama complete! Filled {np.sum(filled)} / {pano_width*pano_height} pixels")
    logging.info(f"Coverage: {100*np.sum(filled)/(pano_width*pano_height):.1f}%")
    return panorama

def save_panorama(panorama, filename='panorama.png'):
    """
    Save the panorama image.
    Parameters:
        panorama: panorama image array
        filename: output filename
    """
    img = Image.fromarray(panorama)
    img.save(filename)
    logging.info(f"Saved panorama to {filename}")

def generate_panorama(q_optimized, imu_ts, cam_file, pano_width=1000, pano_height=500, fov_h=60, fov_v=45, output_file='panorama.png'):
    """
    Generate panorama from optimized quaternions and camera data.
    Parameters:
        q_optimized: optimized orientation quaternions
        imu_ts: IMU timestamps
        cam_file: Path to camera data file
        pano_width: panorama width
        pano_height: panorama height
        fov_h: camera horizontal FOV in degrees
        fov_v: camera vertical FOV in degrees
        output_file: output filename for the panorama
    Returns:
        panorama: generated panorama image
    """
    q_optimized = q_optimized.T
    logging.info("Loading camera data")
    images, cam_ts = load_camera_data(cam_file)
    logging.info(f"Loaded {images.shape[3]} images of size {images.shape[0]}x{images.shape[1]}")
    panorama = create_panorama(images, cam_ts, q_optimized, imu_ts, pano_width, pano_height, fov_h, fov_v)
    save_panorama(panorama, output_file)
    return panorama

def main():
    """
    Main function for generating panorama.
    """
    parser = argparse.ArgumentParser(description='Run calibration on train or test set')
    parser.add_argument('--set', type=str, required=True, choices=['train', 'test'], help='Which dataset to process')
    args = parser.parse_args()
    base_dir = Path(__file__).parent.parent
    save_dir = base_dir / 'results' / args.set
    set_path = f'{args.set}set'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for file_idx in range(1, 12):
        cam_file = base_dir / f'data/{set_path}/cam/cam{file_idx}.p'

        if os.path.exists(str(cam_file)):
            imu_file = base_dir / f'data/{set_path}/imu/imuRaw{file_idx}.p'
            t, accel, gyro = calibrate_imu(imu_file, static_duration=3)
            q_optimized = np.load(save_dir / f'q_optimized_{file_idx}.npy')
            logging.info(f'Generating panorama for file {file_idx}')
            panorama_opt = generate_panorama(q_optimized=q_optimized, imu_ts=t, cam_file=cam_file, pano_width=1000, pano_height=500, fov_h=60, fov_v=45, output_file=f'{save_dir}/panorama_optimized_{file_idx}.png')

if __name__ == '__main__':
    main()
