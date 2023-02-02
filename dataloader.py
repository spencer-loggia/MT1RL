import pandas as pd
from collections import namedtuple
import copy
import torch


def _tensorify_trials(construct_dict, device):
    for key in construct_dict.keys():
        construct_dict[key] = torch.Tensor(construct_dict[key]).long().to(device)
        if len(construct_dict[key].shape) == 1:
            construct_dict[key] = construct_dict[key].reshape(-1, 1)
    return construct_dict


class MTurk1BehaviorData:

    """
    dataloader to run through mturk1 data
    """
    def reindex(self):
        unique_cues = self.data['Cue'].unique()
        unique_targets = self.data['object correct'].unique()
        unique_trial_types = self.data['Task type'].unique()
        cue_reindex = {cue: i for i, cue in enumerate(sorted(unique_cues))}
        target_reindex = {target: i for i, target in enumerate(sorted(unique_targets))}
        trial_type_reindex = {trial: i for i, trial in enumerate(sorted(unique_trial_types))}
        return cue_reindex, target_reindex, trial_type_reindex

    def get_natural_batch(self):
        """
        Grabs batch until a choice is repeated and runs in batch
        :return:
        """
        seen = set()
        construct = {'cue_idx': [], 'choice_options': [], 'choice_made': [], 'correct_option': [], 'trial_type': []}
        for i in range(self.trials_to_load):
            row = self.data.iloc[i]
            cue_idx = row['Cue']
            choice_idx = row['object selected']
            trial_type = row['Task type']
            tup = (cue_idx, trial_type)
            if tup in seen:
                final_batch = _tensorify_trials(copy.copy(construct), device=self.device)
                construct = {'cue_idx': [], 'choice_options': [], 'choice_made': [], 'correct_option': [], 'trial_type': []}
                seen = set()
                seen.add(tup)
                yield final_batch
            seen.add(tup)
            if self.is_4afc:
                construct['cue_idx'].append(cue_idx)
                construct['choice_options'].append([row['choice1'],
                                                    row['choice2'],
                                                    row['choice3'],
                                                    row['choice4']])
                construct['choice_made'].append(choice_idx)
                construct['correct_option'].append(row['object correct'])
                construct['trial_type'].append(row['Task type'])
            else:
                construct['cue_idx'].append(cue_idx)
                construct['choice_options'].append([row['choice1'],
                                                    row['choice2']])
                construct['choice_made'].append(choice_idx)
                construct['correct_option'].append(row['object correct'])
                construct['trial_type'].append(row['Task type'])
        yield _tensorify_trials(construct, device=self.device)

    def __iter__(self):
        for i in range(int(len(self.data))):
            row = self.data.iloc[i]
            self.cue_reindex, self.target_reindex, self.trial_reindex = self.reindex()
            cue_idx = self.cue_reindex[row['Cue']]
            choice_idx = row['object selected']
            if self.is_4afc:
                construct = {'cue_idx': [cue_idx],
                             'choice_options': [[row['choice1'],
                                                 row['choice2'],
                                                 row['choice3'],
                                                 row['choice4']]],
                             'choice_made': [choice_idx],
                             'correct_option': [row['object correct']],
                             'trial_type': [row['Task type']]}
            else:
                construct = {'cue_idx': [cue_idx],
                             'choice_options': [[row['choice1'],
                                                 row['choice2']]],
                             'choice_made': [choice_idx],
                             'correct_option': [row['object correct']],
                             'trial_type': [row['Task type']]}
            yield _tensorify_trials(construct, device=self.device)

    def __repr__(self):
        return self.name

    def __init__(self, path_to_csv, dataset_name, dev='cuda', trials_to_load=200000):
        self.data = pd.read_csv(path_to_csv, index_col='Trial')
        self.is_4afc = False
        if 'choice4' in self.data.columns:
            self.is_4afc = True
        self.trials_to_load = min(len(self.data), trials_to_load)
        if self.is_4afc:
            expected_cols = ('Cue', 'object selected', 'object correct', 'Task type', 'choice1', 'choice2', 'choice3', 'choice4')
        else:
            expected_cols = ('Cue', 'object selected', 'object correct', 'Task type', 'choice1', 'choice2')
        if False in [expc in self.data.columns for expc in expected_cols]:
            raise ValueError("All expected Cols must be in data file passed")
        self.num_cues = 14
        self.num_targets = 14
        self.num_trial_types = self.data['Task type'].nunique()
        self.device = torch.device(dev)
        self.name = str(dataset_name)


if __name__ == '__main__':
    loader = MTurk1BehaviorData(path_to_csv='./data_files/jeeves_probe_no_high_guass.csv')
    for mt1trial in loader:
        print('cue', mt1trial.cue_idx, 'options', mt1trial.choice_options)


