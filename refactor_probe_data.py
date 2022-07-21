import os.path
import sys

import pandas as pd

infile = sys.argv[1]

data = pd.read_csv(infile)

"""
We want the following to have UNIQUE Cues: uncolored shapes, high_gauss_shapes, colors, high gauss colors, achromatic shapes
They can all map to 28 targets w/o overlap 
"""

og_mapping = {"shape_to_color": {"task": 1,
                                         "cues_min_max": (1, 15)},
             "discrimination_shapes_to_color_dist": {"task": 1,
                                       "cues_min_max": (15, 29)},
             "color_to_shape": {"task": 3,
                        "cues_min_max": (1, 15)},
             "color_dist_to_discrimination_shape": {"task": 3,
                                       "cues_min_max": (15, 29)},
             "achromatic_shape_to_shape": {"task": 6,
                                   "cues_min_max": (1, 15)}}
new_task = 1
new_dfs = []
max_cue = 0
for desired_task in og_mapping.keys():
    task_data = data.loc[(data['Task type'] == og_mapping[desired_task]['task']) &
                         (data['Cue'] >= og_mapping[desired_task]['cues_min_max'][0]) &
                         (data['Cue'] < og_mapping[desired_task]['cues_min_max'][1])]
    task_data["Cue"] += max_cue
    max_cue = max(max_cue, max(task_data['Cue']))
    min_target = min(task_data['object correct'])
    task_data['Task type'] = new_task
    task_name_col = [desired_task] * len(task_data)
    task_data['task name'] = task_name_col
    new_dfs.append(task_data)
    new_task += 1

out_dataframe = pd.concat(new_dfs)
out_dataframe.sort_index(inplace=True)
out_dataframe.to_csv(os.path.join(os.path.dirname(infile), "fixed_" + os.path.basename(infile)))


