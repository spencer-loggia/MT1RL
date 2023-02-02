import os.path
import sys
import copy

import pandas as pd

file_base = sys.argv[1]

og_mapping = {"low_gauss_shape_to_color": {"task": (1,),
                                           "cues_min_max": (0, 14)},
              "focal_low_gauss_shape_to_color": {"task": (2,),
                                                 "cues_min_max": (0, 14)},
              "high_gauss_shape_to_color": {"task": (1,),
                                            "cues_min_max": (14, 28)},
              "focal_high_gauss_shape_to_color": {"task": (2,),
                                                  "cues_min_max": (14, 28)},
              "low_gauss_color_to_shape": {"task": (3,),
                                           "cues_min_max": (0, 14)},
              "focal_low_gauss_color_to_shape": {"task": (4,),
                                                 "cues_min_max": (0, 14)},
              "high_gauss_color_to_shape": {"task": (3,),
                                            "cues_min_max": (14, 28)},
              "focal_high_gauss_color_to_shape": {"task": (4,),
                                                  "cues_min_max": (14, 28)},
              "achromatic_shape_to_shape": {"task": (6,),
                                            "cues_min_max": (0, 14)}}

subjects = ['jeeves', 'wooster', 'jocamo']

for subject in subjects:
    fname = subject + file_base
    infile = os.path.join('data_files', fname)
    try:
        data = pd.read_csv(infile)
    except FileNotFoundError:
        print("subject", subject, "not found. skipping...")

    """
    We want the following to have UNIQUE Cues: uncolored shapes, high_gauss_shapes, colors, high gauss colors, achromatic shapes
    They can all map to 28 targets w/o overlap 
    """
    new_task = 0
    new_dfs = []

    for desired_task in og_mapping.keys():
        task_data = data.loc[(data['Task type'].isin(og_mapping[desired_task]['task'])) &
                             (data['Cue'] >= og_mapping[desired_task]['cues_min_max'][0]) &
                             (data['Cue'] < og_mapping[desired_task]['cues_min_max'][1])]
        task_data["Cue state"] = copy.copy(task_data["Cue"])
        task_data["Cue"] -= min(task_data['Cue'])
        task_data[task_data >= 14] -= 14
        task_data['Task type'] = new_task
        task_name_col = [desired_task] * len(task_data)
        task_data['task name'] = task_name_col
        new_dfs.append(task_data)
        new_task += 1

    out_dataframe = pd.concat(new_dfs)
    out_dataframe.set_index('Trial', inplace=True)
    out_dataframe.sort_index(inplace=True)
    out_dataframe.to_csv(os.path.join(os.path.dirname(infile), "fixed_" + os.path.basename(infile)), index=True)
