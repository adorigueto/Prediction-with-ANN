# LIBRARIES
from gc import callbacks
import math
from black import main
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, ShuffleSplit, GroupShuffleSplit
from sklearn import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from functools import partial

# STYLE
# Plot/table options
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
plt.rcParams['font.size'] = 14

# Colors RGB
ccm_black_rgb = [0, 0, 0]
ccm_dgray_rgb = [85, 85, 85]
ccm_gray_rgb = [210, 210, 210]
ccm_dblue_rgb = [25, 50, 120]
ccm_blue_rgb = [20, 80, 200]
ccm_lblue_rgb = [170, 200, 230]
ccm_red_rgb = [190, 0, 0]
ccm_orange_rgb = [255, 100, 0]

colors = [ccm_black_rgb, ccm_dgray_rgb, ccm_gray_rgb, ccm_dblue_rgb, ccm_blue_rgb, ccm_lblue_rgb, ccm_red_rgb,
          ccm_orange_rgb]
color = ['black', 'dgray', 'gray', 'dblue', 'blue', 'lblue', 'red', 'orange']

# Change decimal to binary
i = 0
for i in range(0, 8):
    color[i] = [colors[i] / 255 for colors[i] in colors[i]]

# DATA PREPARATION
# Load file
directory = "C:/Users/André/Documents/00_ITA/00_Mestrado/20_Data_Preparation/"
file_name = "top_force.csv"

file = pd.read_csv(directory + file_name)
print("\nfile head\n", file.head())

# Dropping unnecessary columns
main_df = file.copy()
main_df.drop(['Exp', 'Tool', 'Block', 'SBlock',
           'Position', 'Condition', 'TCond',
           'Length', 'Di', 'Df', 'CTime', 'RAngle'],
           axis=1, inplace=True)
print(main_df.head())

# Stratified train-test split (80/20)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Stratified in relation to the feed rate ('f')
for train_index, test_index in split.split(main_df, main_df['f'], groups=main_df["Run"]):
    stratified_train = main_df.loc[train_index]
    stratified_test = main_df.loc[test_index]

# Check for stratification proportion correctness
print(stratified_train['f'].value_counts() / len(stratified_train))
print(stratified_test['f'].value_counts() / len(stratified_test))

# Check for data leakage --- all ['Run'].value_counts must be == 6
print(stratified_train['Run'].value_counts())

# Define features and labels
features = ['ap', 'vc', 'f', 'Fx', 'Fy', 'Fz', 'F']
labels = ['Ra']

x_train_and_validation = stratified_train.copy()[features]
x_test = stratified_test.copy()[features]

y_train = stratified_train.copy()[labels]
y_test = stratified_test.copy()[labels]

# Spare validation set
x_train, x_val,\
    y_train, y_val = train_test_split(x_train_and_validation, y_train,
    test_size = 0.2, random_state = 42)

# Scale x
sc = preprocessing.StandardScaler(copy=True, with_std=True, with_mean=True)

x_train_sc_np_array = sc.fit_transform(x_train)
x_train_sc = pd.DataFrame(data = x_train_sc_np_array,
    columns = x_train.columns, index = x_train.index)

x_val_sc_np_array = sc.transform(x_val)
x_val_sc = pd.DataFrame(data = x_val_sc_np_array,
    columns = x_val.columns, index=x_val.index)

x_test_sc_np_array = sc.transform(x_test)
x_test_sc = pd.DataFrame(data = x_test_sc_np_array, 
    columns = x_test.columns, index = x_test.index)

# MODEL BUILD
# Clean session
keras.backend.clear_session()

# Define standard layers
Regularized_Dense = partial(keras.layers.Dense, activation = "relu")


# Funtion to create model
def create_model():
    '''This function creates a sequential model'''
    model = keras.Sequential()
    model.add(keras.Input(shape = x_train_sc.shape[1:]))
    model.add(Regularized_Dense(28))
    # model.add(layers.Dropout(0.1))
    model.add(Regularized_Dense(56))
    # model.add(layers.Dropout(0.1))
    model.add(Regularized_Dense(14))
    # model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.95)
    
    model.compile(
        loss = "mean_squared_error",
        optimizer = optimizer,
        metrics = ["mean_absolute_percentage_error", "mean_absolute_error"])
    
    return model


es = EarlyStopping(monitor = "loss",
    min_delta = 0.0001,
    patience = 500,
    verbose = 1,
    mode = "min")

callbacks = [es]

model = create_model()
model.summary()

# LEARN
epochs = 10000
history = model.fit(x_train_sc, y_train, epochs = epochs,
                    validation_data = (x_val_sc, y_val),
                    batch_size = 20, callbacks = callbacks,
                    verbose = 1)

# EVALUATE
loss = model.evaluate(x_test_sc, y_test)

epochs = np.arange(0, len(history.history['loss']))

train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs - 0.5, train_loss, color=color[4], alpha=0.8, label='Training loss')
plt.plot(epochs, val_loss, color=color[7], alpha=0.8, label='Validation loss')
plt.title('Training and Validation loss', size=20, fontweight='bold')
plt.xlabel('Epochs', size=18)
plt.ylabel('Loss', size=18)
plt.legend()
plt.show()