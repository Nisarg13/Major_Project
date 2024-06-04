import csv

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

lambda_factor = 1  # Forgetting factor
learning_rate = 0.01  # Learning rate


def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    """
    Update control torques based on the iterative learning control law.
    """
    errors = desired_angles - current_angles
    control_update = learning_rate * errors

    # Ensure that previous_torques and control_update are of the same size
    new_torques = lambda_factor * previous_torques + control_update
    return new_torques


def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000,
                                    max_time_steps=1000):
    target_angles = np.array([np.pi / 4] * len(actuator_list))  # Target angles for each joint

    error_result = {}  # Dictionary to store trial number as key and (MSE, resultant angle) as value
    data.time = 0
    theta_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    torque_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    previous_torques_series = {trial: np.zeros((max_time_steps, len(actuator_list))) for trial in
                               range(1, num_trials + 1)}
    previous_torques = np.zeros((max_time_steps, len(actuator_list)))
    current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])

    for trial in range(1, num_trials + 1):
        data.qpos[:] = 0  # Reset joint positions to zero at the start of each trial
        # data.ctrl[:] = 0
        if trial <= 400:
            phase = 'BaseLine'
        elif 400 < trial <= 600:
            phase = 'Adaptation'
        elif 600 < trial <= 800:
            phase = 'Washout'
        else:
            phase = 'Readaptation'

        print(f"Starting trial {trial} for {phase} Phase")
        for time_step in range(max_time_steps):
            # if trial == 402:
            #     print('Data_Ctrl: ', data.ctrl)
            #     print('Previous_Torque: ', previous_torques)

            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])
            # if trial == 2:
            #     print(current_angles)

            data.time += model.opt.timestep

            theta_time_series[trial][time_step] = current_angles.tolist()
            # torque_time_series[trial][time_step] = data.ctrl[:len(actuator_list)].tolist()
            previous_torques_series[trial][time_step] = previous_torques[time_step].tolist()
            # data.ctrl[:] = previous_torques[999]

        perturbation = 0
        if phase in ['Adaptation', 'Readaptation']:
            perturbation = -1

        new_torques = np.zeros((max_time_steps, len(actuator_list)))
        # print(previous_torques)
        for i, current_angle in enumerate(theta_time_series[trial].items()):
            # # print(i)

            # print(value[1])
            # Update torques for the entire array

            new_torques[i] = iterative_learning_update(previous_torques[i], target_angles, current_angle[1],
                                                       lambda_factor,
                                                       learning_rate)

            previous_torques[i] = new_torques[i]
            previous_torques_series[trial][i] = previous_torques[i].tolist()
        if trial == 2:
            print(previous_torques_series[trial])

        data.ctrl[0] = new_torques[999][0] + perturbation
        data.ctrl[1] = new_torques[999][1]

        # Update previous torques with the new torques for the next trial

        # print('Trial: ', new_torques)
        error_result[trial] = (
            np.linalg.norm(target_angles - current_angles),
            current_angles.tolist())

    return error_result, theta_time_series, previous_torques_series


# Assuming lambda_factor and learning_rate are defined elsewhere
# Also ensure iterative_learning_update function is defined accordingly to handle the new_torques update logic


def plot_specific_trials_theta_time_series(theta_time_series, trials_to_plot, max_time_steps):
    desired_angle = np.pi / 4  # Desired angle in radians

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in theta_time_series:
            print(f"Trial {trial} data not available.")
            continue

        # Collect the angles for the specified trial and convert to a 2D array
        angles_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if
                       time_step in theta_time_series[trial]]
        angles_array = np.array(angles_list)

        if angles_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        ax1.plot(range(angles_array.shape[0]), angles_array[:, 0], label=f'Theta1 Trial {trial}')
        ax2.plot(range(angles_array.shape[0]), angles_array[:, 1], label=f'Theta2 Trial {trial}')

    ax1.axhline(y=desired_angle, color='r', linestyle='-', label='Desired Angle')
    ax2.axhline(y=desired_angle, color='r', linestyle='-', label='Desired Angle')

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

def plot_error_vs_timesteps(theta_time_series, desired_angle, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in theta_time_series:
            print(f"Trial {trial} data not available.")
            continue

        # Collect the angles for the specified trial and convert to a 2D array
        angles_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if time_step in theta_time_series[trial]]
        angles_array = np.array(angles_list)

        if angles_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        error_theta1 = desired_angle - angles_array[:, 0]
        error_theta2 = desired_angle - angles_array[:, 1]

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

    # print(trials_to_plot)
    for trial in trials_to_plot:
        if trial not in previous_torques_series:
            print(f"Trial {trial} data not available.")
            continue
        # print(previous_torques_series[trial])
        # Collect the torques for the specified trial and convert to a 2D array
        torques_list = [previous_torques_series[trial][time_step] for time_step in range(max_time_steps)]
        # print(torques_list)
        torques_array = np.array(torques_list)

        if torques_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        # print(f"Plotting data for Trial {trial}")
        # print(f"Torque1 for Trial {trial}: {torques_array[:, 0]}")
        # print(f"Torque2 for Trial {trial}: {torques_array[:, 1]}")

        ax1.plot(range(torques_array.shape[0]), torques_array[:, 0], label=f'Torque1 Trial {trial}')
        ax2.plot(range(torques_array.shape[0]), torques_array[:, 1], label=f'Torque2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Torque1')
    ax1.set_title('Torque1 Across Time Steps for Specific Trials')
    ax1.legend()  # Ensure the legend is created

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Torque2')
    ax2.set_title('Torque2 Across Time Steps for Specific Trials')
    ax2.legend()  # Ensure the legend is created

    plt.tight_layout()
    plt.show()


def plot_mse_across_phases(mse_values):
    # Dividing the trials into phases, each with 100 trials
    baseline_trials = mse_values[0:100]
    adaptation_trials = mse_values[400:500]
    washout_trials = mse_values[600:700]
    readaptation_trials = mse_values[800:900]

    # Plotting
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


def write_to_csv(error_result, filepath):
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trial Number', 'MSE', 'Resultant Angles'])
        for trial, (mse, angles) in error_result.items():
            writer.writerow([trial, mse] + angles)


def main():
    model_path = '2R.xml'  # Replace with your actual model path
    actuator_list = ['torque1', 'torque2']  # Replace with your own actuator list
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance *= 5  # Adjust camera distance
        error_result, theta_time_series, previous_torques_series = simulate_with_phases_and_viewer(
            model, data, viewer,
            actuator_list)
        csv_file_path = 'Error_Results.csv'  # Define the path for your CSV file
        write_to_csv(error_result, csv_file_path)
        print(f"Results written to {csv_file_path}")
        # Plotting part remains the same
        trials_to_plot = [1, 50, 100, 150, 200, 250, 300, 350, 400]  # Trials to plot
        plot_specific_trials_theta_time_series(theta_time_series, trials_to_plot, 1000)  # Assuming 1000 time steps

        plot_error_vs_timesteps(theta_time_series, np.pi / 4, trials_to_plot, 1000)  # Plot error vs time steps

        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        plot_mse_across_phases(mse_values)
        plt.stem(trials, mse_values, basefmt=' ')
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Trial Number vs. Norm of Error')
        plt.grid(True)
        plt.show()
        # print(previous_torques_series[400])
        plot_previous_torques_series(previous_torques_series, trials_to_plot, max_time_steps=1000)


if __name__ == "__main__":
    main()

