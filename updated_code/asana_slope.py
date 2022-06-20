#%%
import numpy as np
import pandas as pd
import json
import math
import os

#%%
df = pd.read_csv('train_data.csv')
asanas = ['adho mukha svanasana', 'adho mukha vriksasana', 'bhekasana', 'bhujangasana', 'chakravakasana', 'eka pada koundinyanasana i', 'padmasana', 'simhasana', 'tadasana', 'trikoasana', 'virabhadrasana i', 'virabhadrasana ii', 'virabhadrasana iii']
all_asana_slope = {}
components = ['left arm', 'right arm', 'right knee', 'left knee']
print(df.columns)

#%%
def slope_finder(asana_name):
    left_arm_min, right_arm_min, right_knee_min, left_knee_min = float('inf'), float('inf'), float('inf'), float('inf')
    left_arm_max, right_arm_max, right_knee_max, left_knee_max = float('-inf'), float('-inf'), float('-inf'), float('-inf')
    for i in range(len(df)):
        if asana_name in df.loc[i, "filename"]:
            left_arm_slope = (df['LEFT_SHOULDER_y'][i] - df['LEFT_WRIST_y'][i]) / (df['LEFT_SHOULDER_x'][i] - df['LEFT_WRIST_x'][i])
            if left_arm_slope < left_arm_min:
                left_arm_min = left_arm_slope
            if left_arm_slope > left_arm_max:
                left_arm_max = left_arm_slope
            right_arm_slope = (df['RIGHT_SHOULDER_y'][i] - df['RIGHT_WRIST_y'][i]) / (df['RIGHT_SHOULDER_x'][i] - df['RIGHT_WRIST_x'][i])
            if right_arm_slope < right_arm_min:
                right_arm_min = right_arm_slope
            if right_arm_max < right_arm_slope:
                right_arm_max = right_arm_slope
    slopes = { asana_name: {
        'left_arm_min': left_arm_min,
        'left_arm_max': left_arm_max,
        'right_arm_min': right_arm_min,
        'right_arm_max': right_arm_max
    }}
    return slopes

#%%
for asana in asanas:
    slopes = slope_finder(asana)
    all_asana_slope[asana] = slopes

# %%
print(all_asana_slope)

# %%
with open("slopes.json", "w") as outfile:
    json.dump(all_asana_slope, outfile, indent = 4)
# %%
