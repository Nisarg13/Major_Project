import csv

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

# Constants
LAMBDA_FACTOR = 1  # Forgetting factor
LEARNING_RATE = 0.01  # Learning rate


def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    errors = desired_angles - current_angles
    control_update = learning_rate * errors
    new_torques = lambda_factor * previous_torques + control_update
    return new_torques


def generate_waveform(time_step, max_time_steps):
    t = time_step / max_time_steps  # Normalize time step
    theta1_target = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
    theta2_target = np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t)
    return np.array([theta1_target, theta2_target])


def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000, max_time_steps=1000):
    error_result = {}
    data.time = 0
    theta_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    torque_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    previous_torques = np.zeros((max_time_steps, len(actuator_list)))
    desired_angles = np.array([generate_waveform(time_step, max_time_steps) for time_step in range(max_time_steps)])

    for trial in range(1, num_trials + 1):
        data.qpos[:] = 0  # Reset joint positions to zero at the start of each trial
        phase = get_phase(trial)
        perturbation = get_perturbation(phase)
        print(f"Starting trial {trial} for {phase} Phase")

        for time_step in range(max_time_steps):
            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])
            data.time += model.opt.timestep
            theta_time_series[trial][time_step] = current_angles.tolist()

            target_angles = desired_angles[time_step]
            new_torques = iterative_learning_update(previous_torques[time_step], target_angles, current_angles,
                                                    LAMBDA_FACTOR, LEARNING_RATE)
            previous_torques[time_step] = new_torques
            torque_time_series[trial][time_step] = new_torques.tolist()

            data.ctrl[0] = new_torques[0] + perturbation
            data.ctrl[1] = new_torques[1]

            if time_step % 100 == 0:  # Print every 100 time steps for debugging
                print(
                    f"Trial {trial}, Time step {time_step}, Current Angles: {current_angles}, Target Angles: {target_angles}, New Torques: {new_torques}")

        mse = np.linalg.norm(desired_angles - current_angles)
        error_result[trial] = (mse, current_angles.tolist())

    return error_result, theta_time_series, torque_time_series


def get_phase(trial):
    if trial <= 400:
        return 'BaseLine'
    elif trial <= 600:
        return 'Adaptation'
    elif trial <= 800:
        return 'Washout'
    else:
        return 'Readaptation'


def get_perturbation(phase):
    if phase in ['Adaptation', 'Readaptation']:
        return -1
    return 0


def plot_specific_trials_theta_time_series(theta_time_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    desired_angles = np.array([generate_waveform(time_step, max_time_steps) for time_step in range(max_time_steps)])

    for trial in trials_to_plot:
        if trial not in theta_time_series:
            print(f"Trial {trial} data not available.")
            continue

        angles_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if
                       time_step in theta_time_series[trial]]
        angles_array = np.array(angles_list)

        if angles_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        ax1.plot(range(angles_array.shape[0]), angles_array[:, 0], label=f'Theta1 Actual Trial {trial}')
        ax2.plot(range(angles_array.shape[0]), angles_array[:, 1], label=f'Theta2 Actual Trial {trial}')

    ax1.plot(range(desired_angles.shape[0]), desired_angles[:, 0], '--', label='Theta1 Desired', color='black')
    ax2.plot(range(desired_angles.shape[0]), desired_angles[:, 1], '--', label='Theta2 Desired', color='black')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Theta1 (radians)')
    ax1.set_title('Theta1 Across Time Steps for Specific Trials')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Theta2 (radians)')
    ax2.set_title('Theta2 Across Time Steps for Specific Trials')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_error_vs_timesteps(theta_time_series, desired_angles, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in theta_time_series:
            print(f"Trial {trial} data not available.")
            continue

        angles_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if
                       time_step in theta_time_series[trial]]
        angles_array = np.array(angles_list)

        if angles_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        error_theta1 = desired_angles[:, 0] - angles_array[:, 0]
        error_theta2 = desired_angles[:, 1] - angles_array[:, 1]

        ax1.plot(range(angles_array.shape[0]), error_theta1, label=f'Error Theta1 Trial {trial}')
        ax2.plot(range(angles_array.shape[0]), error_theta2, label=f'Error Theta2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Error Theta1 (radians)')
    ax1.set_title('Error Theta1 Across Time Steps for Specific Trials')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error Theta2 (radians)')
    ax2.set_title('Error Theta2 Across Time Steps for Specific Trials')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_previous_torques_series(previous_torques_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in previous_torques_series:
            print(f"Trial {trial} data not available.")
            continue

        torques_list = [previous_torques_series[trial][time_step] for time_step in range(max_time_steps)]
        torques_array = np.array(torques_list)

        if torques_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        ax1.plot(range(torques_array.shape[0]), torques_array[:, 0], label=f'Torque1 Trial {trial}')
        ax2.plot(range(torques_array.shape[0]), torques_array[:, 1], label=f'Torque2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Torque1')
    ax1.set_title('Torque1 Across Time Steps for Specific Trials')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Torque2')
    ax2.set_title('Torque2 Across Time Steps for Specific Trials')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_mse_across_phases(mse_values):
    baseline_trials = mse_values[0:100]
    adaptation_trials = mse_values[400:500]
    washout_trials = mse_values[600:700]
    readaptation_trials = mse_values[800:900]

    plt.figure(figsize=(10, 5))
    plt.plot(baseline_trials, label='Baseline', linestyle=':', color='black', linewidth=2)
    plt.plot(adaptation_trials, label='Adaptation', linestyle='-', color='red', linewidth=2)
    plt.plot(washout_trials, label='Washout', linestyle='--', color='green', linewidth=2)
    plt.plot(readaptation_trials, label='Readaptation', linestyle='-.', color='blue', linewidth=2)

    plt.xlabel('Trial Number within Phase')
    plt.ylabel('MSE')
    plt.title('MSE Across Different Phases (First 100 Trials)')
    plt.legend()
    plt.grid(True)
    plt.show()


def write_to_csv(error_result, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial', 'Error', 'Current Angles'])
        for trial, (error, angles) in error_result.items():
            writer.writerow([trial, error, angles])


def main():
    model_path = '2R.xml'
    actuator_list = ['torque1', 'torque2']
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 5
        error_result, theta_time_series, torque_time_series = simulate_with_phases_and_viewer(model, data, viewer,
                                                                                              actuator_list)

        csv_file_path = 'Error_Results.csv'
        write_to_csv(error_result, csv_file_path)
        print(f"Results written to {csv_file_path}")

        trials_to_plot = [1, 50, 100, 150, 200, 250, 300, 350, 400]
        plot_specific_trials_theta_time_series(theta_time_series, trials_to_plot, 1000)

        desired_angles = np.array([generate_waveform(time_step, 1000) for time_step in range(1000)])
        plot_error_vs_timesteps(theta_time_series, desired_angles, trials_to_plot, 1000)

        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        plot_mse_across_phases(mse_values)
        plt.stem(trials, mse_values, basefmt=' ')
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Trial Number vs. Norm of Error')
        plt.grid(True)
        plt.show()

        plot_previous_torques_series(torque_time_series, trials_to_plot, max_time_steps=1000)


if __name__ == "__main__":
    main()
