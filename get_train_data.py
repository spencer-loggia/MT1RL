import sys
import os
import json

import numpy as np
import pandas as pd

data_dir = sys.argv[1]
subjects = ['jeeves', 'wooster', 'jocamo']
# trial_descs = ['colored_shape_to_color', 'colored_shape_to_uncolored_shape', 'shape_to_shape', 'color_to_color']
is_2afc = sys.argv[2] == '-2afc'

is_probe = sys.argv[3] == '-probe'
if is_2afc:
    if is_probe:
        task_key = "_2_"
    else:
        task_key = "_1_"

else:
    if is_probe:
        task_key = "_4_"
    else:
        task_key = "_3_"

for subject in subjects:
    base_dfs = []
    for file in sorted(os.listdir(data_dir)):
        if subject in file.lower() and '.txt' in file and task_key in file:
            print(subject, file)
            if is_2afc:
                session_df = pd.DataFrame(columns=['Cue', 'Task type', 'object correct', 'object selected', 'color degree', 'choice1', 'choice2'])
            else:
                session_df = pd.DataFrame(
                    columns=['Cue', 'Task type', 'object correct', 'object selected', 'color degree', 'choice1', 'choice2', 'choice3', 'choice4'])
            with open(os.path.join(data_dir, file), 'r') as f:
                data_dict = json.load(f)[0]
            sample = data_dict['Sample']
            if len(sample) < 5:
                continue
            try:
                resp = np.array(data_dict['Response'])
                good_resp_index = resp != None
                task_type = np.array(data_dict['Routine'])[good_resp_index]
                session_df['Cue'] = np.array(sample)[good_resp_index]
                selected_degrees = [c_options[resp[i]] for i, c_options in enumerate(data_dict['TestC']) if good_resp_index[i]]
                object_selected = [options[data_dict['Response'][i]] for i, options in enumerate(data_dict['Test']) if good_resp_index[i]]
                object_correct = [options[data_dict['CorrectItem'][i]] for i, options in enumerate(data_dict['Test']) if good_resp_index[i]]
                color_degrees = np.array(data_dict['SampleC'])[good_resp_index]
                choices = np.array(list(zip(*data_dict['Test'])))[:, good_resp_index]
            except Exception:
                print(subject, "session #", len(base_dfs), "corrupted. skipping...")
                continue
            session_df['Task type'] = task_type
            session_df['object selected'] = object_selected
            session_df['object correct'] = object_correct
            session_df['color degree'] = color_degrees
            session_df['selected degree'] = selected_degrees
            session_df['choice1'] = choices[0]
            session_df['choice2'] = choices[1]
            if not is_2afc:
                session_df['choice3'] = choices[2]
                session_df['choice4'] = choices[3]
            base_dfs.append(session_df)

    subject_df = pd.concat(base_dfs).reset_index()
    if is_probe:
        desc = 'probe'
    else:
        desc = 'train'
    if is_2afc:
        desc2 = '2afc'
    else:
        desc2 = '4afc'

    subject_df.to_csv(os.path.join('data_files', subject + desc + '_' + desc2 + '_og.csv'), index_label='Trial')





