import csv

import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np

lambda_factor = 1  # Forgetting factor
learning_rate = 0.01  # Learning rate
lambda_pred = 0.7  # Forgetting factor for prediction
gamma_pred = 0.02  # Learning rate for prediction
lambda_ff = 0.7  # Forgetting factor for feedforward control
gamma_ff = 0.02  # Learning rate for feedforward control


def calculate_model_output_error(M_rho, uk_minus_1, y_pred_k_minus_1):
    """
    Calculate the model output error.
    """
    return M_rho @ uk_minus_1 - y_pred_k_minus_1


def iterative_learning_update(previous_torques, desired_angles, current_angles, lambda_factor, learning_rate):
    """
    Update control torques based on the iterative learning control law.
    """
    errors = desired_angles - current_angles
    control_update = learning_rate * errors

    # Ensure that previous_torques and control_update are of the same size
    new_torques = lambda_factor * previous_torques + control_update
    return new_torques, errors


def update_feedforward_control_input(uff, lambda_ff, gamma_ff, M_rho_inv, epsilon_moe):
    return lambda_ff * uff + gamma_ff * M_rho_inv @ np.tanh(epsilon_moe)


def simulate_with_phases_and_viewer(model, data, viewer, actuator_list, num_trials=1000,
                                    max_time_steps=1000):
    target_angles = np.array([np.pi / 4] * len(actuator_list))  # Target angles for each joint

    error_result = {}  # Dictionary to store trial number as key and (MSE, resultant angle) as value
    data.time = 0
    theta_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    previous_torques_series = {trial: np.zeros((max_time_steps, len(actuator_list))) for trial in
                               range(1, num_trials + 1)}
    previous_torques = np.zeros((max_time_steps, len(actuator_list)))
    current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])

    combined_torques = np.zeros((max_time_steps, len(actuator_list)))
    combined_torques_series = {trial: np.zeros((max_time_steps, len(actuator_list))) for trial in
                               range(1, num_trials + 1)}

    error_time_series = {trial: {} for trial in range(1, num_trials + 1)}

    ypred_series = np.zeros_like(current_angles)
    spe_time_series = {trial: {} for trial in range(1, num_trials + 1)}
    ypred_time_series = {trial: {} for trial in range(1, num_trials + 1)}  # Store ypred for each trial and timestep

    uff_time_series = {trial: np.zeros((max_time_steps, len(actuator_list))) for trial in range(1, num_trials + 1)}
    uff = np.zeros((len(actuator_list),))  # Initialize uff for the first time step
    # Initialize M_rho as a 2x2 identity matrix
    M_rho = np.eye(2)
    M_rho_inv = np.linalg.pinv(M_rho)
    print(M_rho_inv)
    # Collect MOE values
    moe_series = {trial: {} for trial in range(1, num_trials + 1)}

    for trial in range(1, num_trials + 1):
        data.qpos[:] = 0  # Reset joint positions to zero at the start of each trial
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
            mujoco.mj_step(model, data)
            viewer.sync()
            current_angles = np.array([data.qpos[i] for i in range(len(actuator_list))])
            data.time += model.opt.timestep

            theta_time_series[trial][time_step] = current_angles.tolist()
            combined_torques_series[trial][time_step] = combined_torques[time_step].tolist()
            previous_torques_series[trial][time_step] = previous_torques[time_step].tolist()

            perturbation = -1 if phase in ['Adaptation', 'Readaptation'] else 0

            model_output_error = calculate_model_output_error(M_rho, combined_torques[time_step], ypred_series)
            moe_series[trial][time_step] = model_output_error.tolist()

            new_torques, errors = iterative_learning_update(previous_torques[time_step], target_angles, current_angles,
                                                            lambda_factor, learning_rate)
            uff = update_feedforward_control_input(
                uff, lambda_ff, gamma_ff, M_rho_inv, model_output_error)
            uff_time_series[trial][time_step] = uff.tolist()

            previous_torques[time_step] = new_torques
            combined_torques[time_step] = new_torques + uff

            previous_torques_series[trial][time_step] = new_torques.tolist()
            combined_torques_series[trial][time_step] = (new_torques + uff).tolist()

            error_time_series[trial][time_step] = errors.tolist()

            data.ctrl[0] = new_torques[0] + perturbation + uff[0]
            data.ctrl[1] = new_torques[1] + uff[1]

            # Calculate SPE
            spe = current_angles - ypred_series
            spe_time_series[trial][time_step] = spe.tolist()

            # Update predicted sensory output
            ypred_series = lambda_pred * ypred_series + gamma_pred * np.tanh(spe)
            ypred_time_series[trial][time_step] = ypred_series.tolist()  # Store ypred

        error_result[trial] = (np.linalg.norm(target_angles - current_angles), current_angles.tolist())
    print(combined_torques_series)
    return (error_result, theta_time_series, previous_torques_series, spe_time_series, ypred_time_series,
            error_time_series, moe_series, uff_time_series, combined_torques_series)


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
        angles_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if
                       time_step in theta_time_series[trial]]
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


def plot_errors_for_trial(error_time_series, trial, time_steps):
    """
    Plot errors for a specific trial and specific time steps.
    """
    if trial not in error_time_series:
        print(f"Trial {trial} data not available.")
        return

    time_steps_available = [ts for ts in time_steps if ts in error_time_series[trial]]
    if not time_steps_available:
        print(f"None of the specified time steps are available for Trial {trial}.")
        return

    errors_list = [error_time_series[trial][ts] for ts in time_steps_available]

    errors_array = np.array(errors_list)

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps_available, errors_array[:, 0], label='Error Theta1')
    plt.plot(time_steps_available, errors_array[:, 1], label='Error Theta2')

    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.title(f'Errors for Trial {trial} across specified Time Steps')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_previous_torques_series(previous_torques_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in previous_torques_series:
            print(f"Trial {trial} data not available.")
            continue

        # Collect the torques for the specified trial and convert to a 2D array
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


def plot_uff_series(uff_time_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in uff_time_series:
            print(f"Trial {trial} data not available.")
            continue

        # Collect the uff values for the specified trial and convert to a 2D array
        uff_list = [uff_time_series[trial][time_step] for time_step in range(max_time_steps)]
        uff_array = np.array(uff_list)

        if uff_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        ax1.plot(range(uff_array.shape[0]), uff_array[:, 0], label=f'uFF1 Trial {trial}')
        ax2.plot(range(uff_array.shape[0]), uff_array[:, 1], label=f'uFF2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('uFF1')
    ax1.set_title('uFF1 Across Time Steps for Specific Trials')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('uFF2')
    ax2.set_title('uFF2 Across Time Steps for Specific Trials')
    ax2.legend()

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


def plot_spe_vs_timesteps(spe_time_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for trial in trials_to_plot:
        if trial not in spe_time_series:
            print(f"Trial {trial} data not available.")
            continue

        spe_list = [spe_time_series[trial][time_step] for time_step in range(max_time_steps) if
                    time_step in spe_time_series[trial]]
        spe_array = np.array(spe_list)

        if spe_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        ax1.plot(range(spe_array.shape[0]), spe_array[:, 0], label=f'SPE Theta1 Trial {trial}')
        ax2.plot(range(spe_array.shape[0]), spe_array[:, 1], label=f'SPE Theta2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('SPE Theta1 (radians)')
    ax1.set_title('SPE Theta1 Across Time Steps for Specific Trials')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('SPE Theta2 (radians)')
    ax2.set_title('SPE Theta2 Across Time Steps for Specific Trials')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_ypred_vs_timesteps(ypred_time_series, theta_time_series, trials_to_plot, max_time_steps):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    colors = plt.cm.get_cmap('tab10', len(trials_to_plot))

    for idx, trial in enumerate(trials_to_plot):
        if trial not in ypred_time_series or trial not in theta_time_series:
            print(f"Trial {trial} data not available.")
            continue

        # Collect ypred values for the specified trial
        ypred_list = [ypred_time_series[trial][time_step] for time_step in range(max_time_steps) if
                      time_step in ypred_time_series[trial]]
        ypred_array = np.array(ypred_list)

        # Collect actual theta values for the specified trial
        theta_list = [theta_time_series[trial][time_step] for time_step in range(max_time_steps) if
                      time_step in theta_time_series[trial]]
        theta_array = np.array(theta_list)

        if ypred_array.shape[0] == 0 or theta_array.shape[0] == 0:
            print(f"No data available for Trial {trial}")
            continue

        color = colors(idx)
        ax1.plot(range(ypred_array.shape[0]), ypred_array[:, 0], label=f'ypred Theta1 Trial {trial}', color=color)
        ax1.plot(range(theta_array.shape[0]), theta_array[:, 0], linestyle='--', color=color,
                 label=f'actual Theta1 Trial {trial}')

        ax2.plot(range(ypred_array.shape[0]), ypred_array[:, 1], label=f'ypred Theta2 Trial {trial}', color=color)
        ax2.plot(range(theta_array.shape[0]), theta_array[:, 1], linestyle='--', color=color,
                 label=f'actual Theta2 Trial {trial}')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Theta1 (radians)')
    ax1.set_title('Theta1 Predictions and Actual Values Across Time Steps')
    ax1.legend()

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Theta2 (radians)')
    ax2.set_title('Theta2 Predictions and Actual Values Across Time Steps')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_moe(moe_series, trials_to_plot):
    """
    Plot Model Output Error (MOE) vs timestamp for specific trials.
    """
    for trial in trials_to_plot:
        if trial not in moe_series:
            print(f"Trial {trial} data not available.")
            continue

        moe_values = moe_series[trial]
        time_steps = list(moe_values.keys())
        moe_values_list = [np.linalg.norm(moe) for moe in moe_values.values()]
        plt.plot(time_steps, moe_values_list, label=f'Trial {trial}')

    plt.xlabel('Time Step')
    plt.ylabel('Model Output Error (MOE)')
    plt.title('Model Output Error (MOE) vs Time Step')
    plt.legend()
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
        error_result, theta_time_series, previous_torques_series, spe_time_series, ypred_time_series, error_time_series, moe_series, uff_time_series, combined_torques_series = simulate_with_phases_and_viewer(
            model, data, viewer,
            actuator_list)
        csv_file_path = 'Error_Results.csv'  # Define the path for your CSV file
        write_to_csv(error_result, csv_file_path)
        print(f"Results written to {csv_file_path}")
        # Plotting part remains the same
        trials_to_plot = [1, 50, 100, 150, 200, 250, 300, 350, 400]  # Trials to plot
        plot_specific_trials_theta_time_series(theta_time_series, trials_to_plot, 1000)  # Assuming 1000 time steps

        plot_error_vs_timesteps(theta_time_series, np.pi / 4, trials_to_plot, 1000)  # Plot error vs time steps

        plot_moe(moe_series, trials_to_plot)
        trials = list(error_result.keys())
        mse_values = [error_result[trial][0] for trial in trials]
        plot_mse_across_phases(mse_values)
        plt.stem(trials, mse_values, basefmt=' ')
        plt.xlabel('Trial Number')
        plt.ylabel('Norm of Error')
        plt.title('Trial Number vs. Norm of Error')
        plt.grid(True)
        plt.show()
        plot_previous_torques_series(previous_torques_series, trials_to_plot, max_time_steps=1000)
        plot_uff_series(uff_time_series, trials_to_plot, max_time_steps=1000)
        plot_spe_vs_timesteps(spe_time_series, trials_to_plot, max_time_steps=1000)
        # Plot the predicted theta values for specific trials
        plot_ypred_vs_timesteps(ypred_time_series, theta_time_series, trials_to_plot, max_time_steps=1000)

        # Get user input for specific trial and time steps
        specific_trial = int(input("Enter the trial number: "))
        specific_time_steps = input("Enter the time steps (comma-separated, e.g., 0,50,100,...): ")
        specific_time_steps = list(map(int, specific_time_steps.split(',')))

        # Plot errors for the user-specified trial and time steps
        plot_errors_for_trial(error_time_series, specific_trial, specific_time_steps)


if __name__ == "__main__":
    main()
