import numpy as np
from transforms3d.quaternions import qmult, qexp 
from scipy.spatial.transform import Rotation
from load_data import read_data
import logging
from plotting import plot_orientation_comparison_train, plot_orientation_comparison_test
from pathlib import Path
import argparse
import os

VREF = 3300
ADC_MAX = 1023
ACCEL_SENS = 300
GYRO_SENS = 3.33 * (180 / np.pi)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calibrate_imu(imu_data, static_duration=3) -> tuple:
    """
    Calibrate IMU data by removing biases and scaling to physical units.
    Parameters:
        imu_data: Path to IMU data file
        static_duration: Duration (in seconds) of the static period at the start (estimated to 3; may need adjustment)
    
    Returns:
        t: Time vector
        accel: Acceleration data
        gyro: Angular velocity data
    """
    imu_data = read_data(imu_data)
    t = imu_data[0]
    accel_raw = imu_data[1:4]
    gyro_raw = imu_data[4:7]
    accel_scale = VREF / ADC_MAX / ACCEL_SENS
    gyro_scale = VREF / ADC_MAX / GYRO_SENS
    static_mask = t < (t[0] + static_duration)
    accel_bias = np.mean(accel_raw[:, static_mask], axis=1)
    gyro_bias = np.mean(gyro_raw[:, static_mask], axis=1)
    accel = (accel_raw - accel_bias[:, None]) * accel_scale
    gyro = (gyro_raw - gyro_bias[:, None]) * gyro_scale
    accel += (np.array([0, 0, 1]) - np.mean(accel[:, static_mask], axis=1))[:, None]
    gyro -= np.mean(gyro[:, static_mask], axis=1)[:, None]
    return t, accel, gyro

def integrate_gyro(t, gyro) -> np.ndarray:
    """
    Integrate gyroscope data to obtain orientation quaternions.
    Parameters:
        t: Time vector
        gyro: Angular velocity data
    Returns:
        q: Orientation quaternions
    """
    N = t.size
    q = np.zeros((4, N))
    q[:, 0] = [1, 0, 0, 0]

    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        dq = qexp(np.hstack(([0], 0.5 * dt * gyro[:, k])))
        q[:, k + 1] = qmult(q[:, k], dq)
        q[:, k + 1] /= np.linalg.norm(q[:, k + 1])

    return q

def quaternions_to_euler(q) -> np.ndarray:
    """
    Convert orientation quaternions to Euler angles (roll, pitch, yaw).
    Parameters:
        q: Orientation quaternions
    Returns:
        rpy: Euler angles (roll, pitch, yaw)
    """
    N = q.shape[1]
    rpy = np.zeros((N, 3))
    
    for k in range(N):
        r = Rotation.from_quat([q[1, k], q[2, k], q[3, k], q[0, k]])
        rpy[k] = r.as_euler('xyz', degrees=False)
    
    return np.unwrap(rpy, axis=0)

def vicon_to_euler(vicon_data) -> tuple:
    """
    Convert Vicon ground truth rotation matrices to Euler angles (roll, pitch, yaw).
    Parameters:
        vicon_data: Path to Vicon data file
    Returns:
        ts: Time vector
        rpy: Euler angles (roll, pitch, yaw)
    """
    vicon_data = read_data(vicon_data)
    rots = vicon_data['rots']
    ts = vicon_data['ts'].flatten()
    valid_mask = ~np.isnan(rots[0, 0, :])
    valid_indices = np.where(valid_mask)[0]
    N_valid = len(valid_indices)
    rpy = np.zeros((N_valid, 3))
    
    for i, k in enumerate(valid_indices):
        r = Rotation.from_matrix(rots[:, :, k])
        rpy[i] = r.as_euler('xyz', degrees=False)
    
    return ts[valid_indices], np.unwrap(rpy, axis=0)

def main():
    """
    Main function to run orientation tracking for IMU data before PGD
    """
    parser = argparse.ArgumentParser(description='Run calibration on train or test set')
    parser.add_argument('--set', type=str, required=True, choices=['train', 'test'], help='Which dataset to process')
    args = parser.parse_args()
    train = (args.set == 'train')
    base_dir = Path(__file__).parent.parent
    save_dir = base_dir / 'results' / args.set
    set_path = f'{args.set}set'
    save_dir.mkdir(parents=True, exist_ok=True)

    for file_idx in range(1, 12):
        imu_file = base_dir / 'data' / set_path / 'imu' / f'imuRaw{file_idx}.p'

        if not os.path.exists(str(imu_file)):
            continue
    
        logging.info(f'Calibrating IMU for file {file_idx}')

        if train:
            logging.info('Loading VICON data')
            vicon_file = base_dir / 'data' / set_path / 'vicon' / f'viconRot{file_idx}.p'
            vicon_ts, vicon_rpy = vicon_to_euler(vicon_file)

        t, accel, gyro = calibrate_imu(imu_file, static_duration=3)
        q_simple = integrate_gyro(t, gyro)
        save_path = save_dir / f'q_simple_{file_idx}.npy'   
        np.save(save_path, q_simple)
        imu_rpy_simple = quaternions_to_euler(q_simple)
            
        if train:
            logging.info('Plotting simple integration vs VICON')
            plot_orientation_comparison_train(t, imu_rpy_simple[:, 0], imu_rpy_simple[:, 1], imu_rpy_simple[:, 2], 'IMU', vicon_ts, vicon_rpy[:, 0], vicon_rpy[:, 1], vicon_rpy[:, 2], 'VICON', f'{save_dir}/simple_integration_comparison_{file_idx}.png', title=f'Simple Integration vs VICON Ground Truth for File {file_idx}')
        else:
            logging.info('Plotting simple integration results')
            plot_orientation_comparison_test(t, imu_rpy_simple[:, 0], imu_rpy_simple[:, 1], imu_rpy_simple[:, 2], f'{save_dir}/simple_integration_results_{file_idx}.png', title=f'Simple Integration Results for File {file_idx}')
            
if __name__ == "__main__":
    main()
