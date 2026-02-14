import numpy as np
import matplotlib.pyplot as plt

def plot_orientation_comparison_train(t, imu_roll, imu_pitch, imu_yaw, label1, vicon_ts, vicon_roll, vicon_pitch, vicon_yaw, label2, filename, title='IMU vs VICON'):
    """
    Plot comparison of IMU and VICON orientation estimates.
    Parameters:
        t: Time vector for IMU data
        imu_roll: Roll angles from IMU
        imu_pitch: Pitch angles from IMU
        imu_yaw: Yaw angles from IMU
        vicon_ts: Time vector for VICON data
        vicon_roll: Roll angles from VICON
        vicon_pitch: Pitch angles from VICON
        vicon_yaw: Yaw angles from VICON
        title: Title of the plot
        filename: Filename to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    imu_roll_deg = np.degrees(imu_roll)
    imu_pitch_deg = np.degrees(imu_pitch)
    imu_yaw_deg = np.degrees(imu_yaw)
    vicon_roll_deg = np.degrees(vicon_roll)
    vicon_pitch_deg = np.degrees(vicon_pitch)
    vicon_yaw_deg = np.degrees(vicon_yaw)

    # Roll
    axes[0].plot(t, imu_roll_deg, 'r-', linewidth=2, label=label1, alpha=0.8)
    axes[0].plot(vicon_ts, vicon_roll_deg, 'b-', linewidth=1.5, label=label2, alpha=0.7)
    axes[0].set_ylabel('Roll (deg)', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Pitch
    axes[1].plot(t, imu_pitch_deg, 'r-', linewidth=2, label=label1, alpha=0.8)
    axes[1].plot(vicon_ts, vicon_pitch_deg, 'b-', linewidth=1.5, label=label2, alpha=0.7)
    axes[1].set_ylabel('Pitch (deg)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Yaw
    axes[2].plot(t, imu_yaw_deg, 'r-', linewidth=2, label=label1, alpha=0.8)
    axes[2].plot(vicon_ts, vicon_yaw_deg, 'b-', linewidth=1.5, label=label2, alpha=0.7)
    axes[2].set_ylabel('Yaw (deg)', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{filename}'")

def plot_orientation_comparison_test(t, imu_roll, imu_pitch, imu_yaw, filename, title='IMU Calibration Results'):
    """
    Plot comparison of IMU and VICON orientation estimates.
    Parameters:
        t: Time vector for IMU data
        imu_roll: Roll angles from IMU
        imu_pitch: Pitch angles from IMU
        imu_yaw: Yaw angles from IMU
        title: Title of the plot
        filename: Filename to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    imu_roll_deg = np.degrees(imu_roll)
    imu_pitch_deg = np.degrees(imu_pitch)
    imu_yaw_deg = np.degrees(imu_yaw)

    # Roll
    axes[0].plot(t, imu_roll_deg, 'r-', linewidth=2, label='IMU', alpha=0.8)
    axes[0].set_ylabel('Roll (deg)', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Pitch
    axes[1].plot(t, imu_pitch_deg, 'r-', linewidth=2, label='IMU', alpha=0.8)
    axes[1].set_ylabel('Pitch (deg)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Yaw
    axes[2].plot(t, imu_yaw_deg, 'r-', linewidth=2, label='IMU', alpha=0.8)
    axes[2].set_ylabel('Yaw (deg)', fontsize=12)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{filename}'")
