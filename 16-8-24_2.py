import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import datetime

# Define the number of time samples used for training and the discretization step (sampling)
time = 600  # Adjust this according to your data

current_date = datetime.datetime.now().strftime("%Y%m%d")
###############################################################################
#                   Load the data
###############################################################################
# Load torque input sequences (these are now the inputs to the model)
input_seq_test = pd.read_csv('test_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
output_test = pd.read_csv('test_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]

# Reshape the input and output data for the LSTM model
input_seq_test = np.reshape(input_seq_test, (1, input_seq_test.shape[0], input_seq_test.shape[1]))
output_test = np.reshape(output_test, (1, output_test.shape[0], output_test.shape[1]))

###############################################################################
#                Load the model and make predictions
###############################################################################
# Load the model
model = load_model('lstm_model.h5')

# Load the history (if needed)
history = joblib.load('history.pkl')

# Use the test data to predict the model response (predicted angles)
testPredict = model.predict(input_seq_test)

###############################################################################
#  Plot the predicted and "true" output and plot training and validation losses
###############################################################################

# Plot the predicted angles vs. the true angles from the test data
time_plot = range(1, time + 1)

# Plot the predicted vs. true angles for Angle 1
plt.figure()
plt.plot(time_plot, testPredict[0, :, 0], label='Predicted Angle 1')
plt.plot(time_plot, output_test[0, :, 0], 'r', label='True Angle 1')
plt.xlabel('Discrete time steps')
plt.ylabel('Angle 1')
plt.legend()
plt.title('Predicted vs. True Angle 1')
angle1_plot_filename = f'responseLSTM32_Angle1_{current_date}.png'
plt.savefig(angle1_plot_filename)
plt.show()

# Plot the predicted vs. true angles for Angle 2
plt.figure()
plt.plot(time_plot, testPredict[0, :, 1], label='Predicted Angle 2')
plt.plot(time_plot, output_test[0, :, 1], 'r', label='True Angle 2')
plt.xlabel('Discrete time steps')
plt.ylabel('Angle 2')
plt.legend()
plt.title('Predicted vs. True Angle 2')
angle2_plot_filename = f'responseLSTM32_Angle2_{current_date}.png'
plt.savefig(angle2_plot_filename)
plt.show()

# Plot the training and validation losses
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
plt.legend()
loss_plot_filename = f'lossLSTM32_{current_date}.png'
plt.savefig(loss_plot_filename)
plt.show()
