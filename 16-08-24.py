# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM, TimeDistributed
# from keras.optimizers import RMSprop
# import joblib
#
# # Define the number of time samples used for training and the discretization step (sampling)
# time = 600  # Adjust this according to your data
#
# ###############################################################################
# #                   Load the data
# ###############################################################################
# # Load torque input sequences (these are now the inputs to the model)
# input_seq_train = pd.read_csv('train_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
# input_seq_validate = pd.read_csv('val_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
# input_seq_test = pd.read_csv('test_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
#
# # Load the corresponding angle output sequences (these are now the targets)
# output_train = pd.read_csv('train_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]
# output_validate = pd.read_csv('val_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]
# output_test = pd.read_csv('test_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]
#
# # Reshape the input and output data for the LSTM model
# input_seq_train = np.reshape(input_seq_train, (1, input_seq_train.shape[0], input_seq_train.shape[1]))
# input_seq_validate = np.reshape(input_seq_validate, (1, input_seq_validate.shape[0], input_seq_validate.shape[1]))
# input_seq_test = np.reshape(input_seq_test, (1, input_seq_test.shape[0], input_seq_test.shape[1]))
#
# output_train = np.reshape(output_train, (1, output_train.shape[0], output_train.shape[1]))
# output_validate = np.reshape(output_validate, (1, output_validate.shape[0], output_validate.shape[1]))
# output_test = np.reshape(output_test, (1, output_test.shape[0], output_test.shape[1]))
#
# ###############################################################################
# #                Here we define the network
# ###############################################################################
# # Define the LSTM model
# model = Sequential()
# # First LSTM layer
# model.add(LSTM(128, input_shape=(input_seq_train.shape[1], input_seq_train.shape[2]), return_sequences=True))
# # Second LSTM layer
# model.add(LSTM(256, return_sequences=True))
# # Third LSTM layer
# model.add(LSTM(16, return_sequences=True))
# # TimeDistributed Dense layer for output
# model.add(TimeDistributed(Dense(8, activation='relu')))
# # Output layer
# model.add(TimeDistributed(Dense(2)))  # Output 2 angles at each time step
#
# model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])
#
# # Train the model using torque inputs to predict angles
# history = model.fit(input_seq_train, output_train, epochs=2000, batch_size=1, validation_data=(input_seq_validate, output_validate), verbose=2)
#
# # Save the model
# model.save('lstm_model.h5')
#
# # Optionally, save the history object using joblib
# joblib.dump(history.history, 'history.pkl')
#
# # Use the test data to predict the model response (predicted angles)
# testPredict = model.predict(input_seq_test)
#
# ###############################################################################
# #  Plot the predicted and "true" output and plot training and validation losses
# ###############################################################################
#
# # Plot the predicted angles vs. the true angles from the test data
# time_plot = range(1, time + 1)
# # Plot the predicted vs. true angles for Angle 1
# plt.figure()
# plt.plot(time_plot, testPredict[0, :, 0], label='Predicted Angle 1')
# plt.plot(time_plot, output_test[0, :, 0], 'r', label='True Angle 1')
# plt.xlabel('Discrete time steps')
# plt.ylabel('Angle 1')
# plt.legend()
# plt.title('Predicted vs. True Angle 1')
# plt.savefig('responseLSTM32_Angle1.png')
# plt.show()
#
# # Plot the predicted vs. true angles for Angle 2
# plt.figure()
# plt.plot(time_plot, testPredict[0, :, 1], label='Predicted Angle 2')
# plt.plot(time_plot, output_test[0, :, 1], 'r', label='True Angle 2')
# plt.xlabel('Discrete time steps')
# plt.ylabel('Angle 2')
# plt.legend()
# plt.title('Predicted vs. True Angle 2')
# plt.savefig('responseLSTM32_Angle2.png')
# plt.show()
#
# # Plot the training and validation losses
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation losses')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.xscale('log')
# plt.legend()
# plt.savefig('lossLSTM32.png')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, TimeDistributed
from keras.optimizers import RMSprop
import joblib
import datetime

# Define the number of time samples used for training and the discretization step (sampling)
time = 600  # Adjust this according to your data

# Get the current date in the format YYYYMMDD
current_date = datetime.datetime.now().strftime("%Y%m%d")

###############################################################################
#                   Load the data
###############################################################################
# Load torque input sequences (these are now the inputs to the model)
input_seq_train = pd.read_csv('train_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
input_seq_validate = pd.read_csv('val_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]
input_seq_test = pd.read_csv('test_input_torques.csv')[['Torque 1', 'Torque 2']].values[:time]

# Load the corresponding angle output sequences (these are now the targets)
output_train = pd.read_csv('train_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]
output_validate = pd.read_csv('val_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]
output_test = pd.read_csv('test_output_angles.csv')[['Angle 1', 'Angle 2']].values[:time]

# Reshape the input and output data for the LSTM model
input_seq_train = np.reshape(input_seq_train, (1, input_seq_train.shape[0], input_seq_train.shape[1]))
input_seq_validate = np.reshape(input_seq_validate, (1, input_seq_validate.shape[0], input_seq_validate.shape[1]))
input_seq_test = np.reshape(input_seq_test, (1, input_seq_test.shape[0], input_seq_test.shape[1]))

output_train = np.reshape(output_train, (1, output_train.shape[0], output_train.shape[1]))
output_validate = np.reshape(output_validate, (1, output_validate.shape[0], output_validate.shape[1]))
output_test = np.reshape(output_test, (1, output_test.shape[0], output_test.shape[1]))

###############################################################################
#                Here we define the network
###############################################################################
# Define the LSTM model
model = Sequential()
# First LSTM layer
model.add(LSTM(128, input_shape=(input_seq_train.shape[1], input_seq_train.shape[2]), return_sequences=True))
# Second LSTM layer
model.add(LSTM(256, return_sequences=True))
# Third LSTM layer
model.add(LSTM(16, return_sequences=True))
# TimeDistributed Dense layer for output
model.add(TimeDistributed(Dense(8, activation='relu')))
# Output layer
model.add(TimeDistributed(Dense(2)))  # Output 2 angles at each time step

model.compile(optimizer=RMSprop(), loss='mean_squared_error', metrics=['mse'])

# Train the model using torque inputs to predict angles
history = model.fit(input_seq_train, output_train, epochs=2000, batch_size=1, validation_data=(input_seq_validate, output_validate), verbose=2)

# Save the model
model_filename = f'lstm_model_{current_date}.h5'
model.save(model_filename)

# Optionally, save the history object using joblib
history_filename = f'history_{current_date}.pkl'
joblib.dump(history.history, history_filename)

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
loss = history.history['loss']
val_loss = history.history['val_loss']
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
