# LIBRARIES
import math
from black import main
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
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
           'Length', 'Di', 'Df', 'CTime', 'RAngle', 'Run'],
           axis=1, inplace=True)
print(main_df.head())

# Stratified train-test split (80/20)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Stratified in relation to the feed rate ('f')
for train_index, test_index in split.split(main_df, main_df['f']):
    stratified_train = main_df.loc[train_index]
    stratified_test = main_df.loc[test_index]

# Check for stratification correctness
print(stratified_train['f'].value_counts() / len(stratified_train))
print(stratified_test['f'].value_counts() / len(stratified_test))

# Define features and labels
features = ['ap', 'vc', 'f', 'Fx', 'Fy', 'Fy', 'Fz', 'F']
labels = ['Ra']

x_train_and_validation = stratified_train.copy()[features]
x_test = stratified_test.copy()[features]

y_train = stratified_train.copy()[labels]
y_test = stratified_test.copy()[labels]

