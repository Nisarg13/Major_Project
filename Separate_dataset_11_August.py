import csv
import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

def write_torques_to_csv(torques, model, csv_filename):
    """Write the collected torques to a CSV file."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Time Step'] + [f'Torque {i+1}' for i in range(model.nu)]
        writer.writerow(headers)
        for i, torque in enumerate(torques):
            writer.writerow([i] + torque.tolist())

def write_angles_to_csv(angles, model, csv_filename):
    """Write the collected angles to a CSV file."""
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Time Step'] + [f'Angle {i+1}' for i in range(model.nq)]
        writer.writerow(headers)
        for i, angle in enumerate(angles):
            writer.writerow([i] + angle.tolist())

def generate_torques(model, data, max_wall_time=100, noise_scale=0.05, seed=42):
    np.random.seed(seed)
    start_time = time.time()
    torques = []

    while time.time() - start_time < max_wall_time:
        step_start_time = time.time()

        ar1 = np.random.uniform(-10, 10)
        ar2 = np.random.uniform(-10, 10)
        ar5 = np.random.uniform(-10, 10)
        ar7 = np.random.uniform(-10, 10)
        fr3 = np.random.uniform(0.001, 0.1)
        fr4 = np.random.uniform(0.001, 0.1)
        fr6 = np.random.uniform(0.001, 0.1)
        fr8 = np.random.uniform(0.001, 0.1)
        t = time.time() - start_time

        t1 = ar1 * np.sin(2 * np.pi * fr3 * t) + ar2 * np.cos(2 * np.pi * fr4 * t) + noise_scale * np.random.randn()
        t2 = ar5 * np.cos(2 * np.pi * fr6 * t) + ar7 * np.sin(2 * np.pi * fr8 * t) + noise_scale * np.random.randn()

        data.ctrl[:] = [t1, t2]
        torques.append(data.ctrl.copy())

        mujoco.mj_step(model, data)

        time_until_next_step = model.opt.timestep - (time.time() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    return torques

def simulate_with_torques(model, data, torques, viewer):
    angles = []
    for i, torque in enumerate(torques):
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        viewer.sync()
        angles.append(data.qpos.copy())
    return angles

def generate_and_save_data(model, data, viewer, dataset_type, max_wall_time, noise_scale, seed):
    torques = generate_torques(model, data, max_wall_time=max_wall_time, noise_scale=noise_scale, seed=seed)
    write_torques_to_csv(torques, model, f'{dataset_type}_input_torques.csv')

    angles = simulate_with_torques(model, data, torques, viewer)
    write_angles_to_csv(angles, model, f'{dataset_type}_output_angles.csv')

    print(f"{dataset_type.capitalize()} data generated and saved.")

def main():
    model_path = '2R.xml'
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    max_wall_time = 100
    noise_scale = 0.05

    with mujoco.viewer.launch_passive(model, data) as viewer:
        generate_and_save_data(model, data, viewer, 'train', max_wall_time, noise_scale, seed=42)
        generate_and_save_data(model, data, viewer, 'val', max_wall_time, noise_scale, seed=43)
        generate_and_save_data(model, data, viewer, 'test', max_wall_time, noise_scale, seed=44)

if __name__ == "__main__":
    main()
