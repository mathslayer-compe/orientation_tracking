import numpy as np
import torch
from imu_utils import integrate_gyro, calibrate_imu, vicon_to_euler, quaternions_to_euler
from plotting import plot_orientation_comparison_train
import logging
from pathlib import Path
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def quat_multiply_torch(q1, q2) -> torch.Tensor:
    """
    Multiply two quaternions using PyTorch.
    Parameters:
        q1: First quaternion
        q2: Second quaternion
    Returns:
        result: Resulting quaternion from multiplication
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_inverse_torch(q) -> torch.Tensor:
    """
    Compute the inverse of a quaternion using PyTorch.
    Parameters:
        q: Input quaternion
    Returns: 
        result: Inverse of the input quaternion
    """
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quat_exp_torch(v) -> torch.Tensor:
    """
    Exponential map for quaternions using PyTorch.
    Parameters:
        v: Axis-angle representation
    Returns:
        q: Resulting quaternion
    """
    theta = torch.norm(v, dim=-1)
    small_angle = (theta < 1e-8).squeeze(-1)
    half_theta = theta / 2
    sinc_half = torch.where(small_angle, torch.ones_like(theta.squeeze(-1)) - (half_theta.squeeze(-1)**2) / 6, torch.sin(half_theta.squeeze(-1)) / theta.squeeze(-1))
    w = torch.cos(half_theta.squeeze(-1))
    xyz = v * sinc_half.unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)

def quat_log_torch(q) -> torch.Tensor:
    """
    Logarithmic map for quaternions using PyTorch.
    Parameters:
        q: Input quaternion
    Returns:    
        axis_angle: Axis-angle representation
    """
    w = q[..., 0]
    v = q[..., 1:4]
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    small_rotation = (v_norm.squeeze(-1) < 1e-8)
    theta = 2 * torch.atan2(v_norm.squeeze(-1), w)
    axis = torch.where(small_rotation.unsqueeze(-1), v, v / v_norm)
    return axis * theta.unsqueeze(-1)

def motion_model_torch(q_t, omega_t, dt) -> torch.Tensor:
    """
    Motion model for quaternion propagation using gyroscope data to be workable in PyTorch.
    Parameters:
        q_t: Current orientation quaternion
        omega_t: Angular velocity
        dt: Time step
    Returns:
        q_t1: Predicted orientation quaternion at next time step
    """
    axis_angle = dt * omega_t / 2
    delta_q = quat_exp_torch(axis_angle)
    return quat_multiply_torch(q_t, delta_q)

def observation_model_torch(q_t) -> torch.Tensor:
    """
    Observation model to predict accelerometer measurements from orientation quaternions using PyTorch.
    Parameters:
        q_t: Orientation quaternion
    Returns:
        predicted_accel: Predicted accelerometer measurements
    """
    gravity = torch.zeros_like(q_t)
    gravity[..., 3] = 1.0
    q_inv = quat_inverse_torch(q_t)
    temp = quat_multiply_torch(q_inv, gravity)
    predicted_accel = quat_multiply_torch(temp, q_t)
    return predicted_accel

def cost_function(q_traj, gyro, accel, ts) -> torch.Tensor:
    """"
    Compute the total cost for a trajectory of quaternions.
    Parameters:
        q_traj: Trajectory of quaternions
        gyro: Gyroscope measurements
        accel: Accelerometer measurements
        ts: Timestamps
    Returns:
        total_cost: Total cost value
    """
    T = len(ts)
    motion_cost = torch.tensor(0.0, device=q_traj.device)
    observation_cost = torch.tensor(0.0, device=q_traj.device)
    
    for t in range(T - 1):
        dt = ts[t + 1] - ts[t]
        q_predicted = motion_model_torch(q_traj[t], gyro[t], dt)
        q_error = quat_multiply_torch(quat_inverse_torch(q_traj[t + 1]), q_predicted)
        log_error = quat_log_torch(q_error)
        motion_cost = motion_cost + torch.sum((2 * log_error)**2)
    
    for t in range(T):
        predicted_accel = observation_model_torch(q_traj[t])
        measured_accel = torch.cat([torch.zeros(1, device=accel.device), accel[t]])
        error = measured_accel - predicted_accel
        observation_cost = observation_cost + torch.sum(error**2)
    
    total_cost = 0.5 * motion_cost + 0.5 * observation_cost
    return total_cost

def optimize_orientation(t, gyro, accel, max_iters=300, lr=1e-2) -> tuple:
    """
    Optimize orientation quaternions using Projected Gradient Descent (PGD).
    Parameters:
        t: Time vector
        gyro: Gyroscope measurements
        accel: Accelerometer measurements
        max_iters: Maximum number of iterations
        lr: Learning rate
    Returns:
        q_optimized: Optimized orientation quaternions
        costs: List of cost values over iterations
    """
    T = len(t)
    q_init = integrate_gyro(t, gyro).T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_traj = torch.tensor(q_init, dtype=torch.float32, device=device, requires_grad=True)
    gyro_torch = torch.tensor(gyro.T, dtype=torch.float32, device=device)
    accel_torch = torch.tensor(accel.T, dtype=torch.float32, device=device)
    ts_torch = torch.tensor(t, dtype=torch.float32, device=device)
    costs = []
    logging.info(f"Starting PGD optimization for {max_iters} iterations...")
    
    for i in range(max_iters):
        if q_traj.grad is not None:
            q_traj.grad.zero_()

        cost = cost_function(q_traj, gyro_torch, accel_torch, ts_torch)
        cost.backward()
        
        with torch.no_grad():
            q_traj -= lr * q_traj.grad
            q_traj /= torch.norm(q_traj, dim=1, keepdim=True)
        
        costs.append(cost.item())
        
        if i % 10 == 0 or i == max_iters - 1:
            logging.info(f"Iter {i:4d}: Cost = {cost.item():12.6f}")
        
        if i > 20:
            rel_improvement = abs(costs[-2] - costs[-1]) / max(costs[-2], 1e-8)

            if rel_improvement < 1e-6:
                logging.info(f"Converged at iteration {i}")
                break
    
    logging.info("Optimization complete!")
    q_optimized = q_traj.detach().cpu().numpy()
    return q_optimized, costs

def main():
    """
    Main function to run orientation optimization on training dataset.
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
        logging.info(f'Optimizing trajectory for file {file_idx}')
        q_opt, costs = optimize_orientation(t, gyro, accel)
        imu_rpy_opt = quaternions_to_euler(q_opt.T)
        save_path = save_dir / f'q_optimized_{file_idx}.npy'
        np.save(save_path, q_opt)

        if train:
            vicon_ts, vicon_rpy = vicon_to_euler(vicon_file)
            plot_orientation_comparison_train(t, imu_rpy_opt[:,0], imu_rpy_opt[:,1], imu_rpy_opt[:,2], 'IMU', vicon_ts, vicon_rpy[:,0], vicon_rpy[:,1], vicon_rpy[:,2], 'VICON', f'{save_dir}/optimized_comparison_{file_idx}.png', title=f'Optimized IMU Measurements vs VICON Ground Truth for File {file_idx}')
        else:
            q_simple = np.load(save_dir / f'q_simple_{file_idx}.npy')
            imu_rpy_simple = quaternions_to_euler(q_simple)
            plot_orientation_comparison_train(t, imu_rpy_opt[:,0], imu_rpy_opt[:,1], imu_rpy_opt[:,2], 'Optimized IMU Measurements', t, imu_rpy_simple[:,0], imu_rpy_simple[:,1], imu_rpy_simple[:,2], 'Simple IMU Integration Measurements', f'{save_dir}/optimized_comparison_{file_idx}.png', title=f'Optimized vs Simple Integration for File {file_idx}')

    logging.info('Finished optimizing all sequences')

if __name__ == '__main__':
    main()
