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
    def reindex(self, ignore_tasks):
        index_map = {}
        unique_trial_types = self.data['Task type'].unique()
        trial_type_reindex = {trial: i for i, trial in enumerate(sorted(unique_trial_types)) if trial not in ignore_tasks}
        for key in trial_type_reindex.keys():
            idx = trial_type_reindex[key]
            index_map[idx] = {}
            trial_data = self.data.loc[self.data['Task type'] == key]
            unique_cues = trial_data['Cues'].unique()
            unique_targets = trial_data['object correct'].unique()
            index_map[idx]['cues'] = {cue: i for i, cue in enumerate(sorted(unique_cues))}
            index_map[idx]['targets'] = {target: i for i, target in enumerate(sorted(unique_targets))}

        return trial_type_reindex, index_map

    def get_natural_batch(self):
        """
        Grabs batch until a choice is repeated and runs in batch
        :return:
        """
        cues = set()
        choice = set()
        construct = {'cue_idx': [], 'choice_options': [], 'choice_made': [], 'correct_option': [], 'trial_type': []}
        for i in range(55000):
            row = self.data.iloc[i]
            task_type = self.trial_type_reindex_map[row['Task type']]
            cue_reindex_map = self.index_map[task_type]['cues']
            target_reindex_map = self.index_map[task_type]['targets']
            cue_idx = cue_reindex_map[row['Cue']]
            choice_idx = target_reindex_map[row['object selected']]
            if cue_idx in cues and choice_idx in choice:
                final_batch = _tensorify_trials(copy.copy(construct), device=self.device)
                construct = {'cue_idx': [], 'choice_options': [], 'choice_made': [], 'correct_option': [], 'trial_type': []}
                cues = set()
                choice = set()
                yield final_batch
            cues.add(cue_idx)
            choice.add(choice_idx)
            construct['cue_idx'].append(cue_idx)
            construct['choice_options'].append([target_reindex_map[row['choice1']],
                                                target_reindex_map[row['choice2']],
                                                target_reindex_map[row['choice3']],
                                                target_reindex_map[row['choice4']]])
            construct['choice_made'].append(choice_idx)
            construct['correct_option'].append(target_reindex_map[row['object correct']])
            construct['trial_type'].append(task_type)
        yield _tensorify_trials(construct, device=self.device)

    def __iter__(self):
        for i in range(int(len(self.data))):
            row = self.data.iloc[i]
            cue_reindex_map = self.index_map[task_type]['cues']
            target_reindex_map = self.index_map[task_type]['targets']
            cue_idx = cue_reindex_map[row['Cue']]
            choice_idx = target_reindex_map[row['object selected']]
            construct = {'cue_idx': [cue_idx],
                         'choice_options': [[target_reindex_map[row['choice1']],
                                                target_reindex_map[row['choice2']],
                                                target_reindex_map[row['choice3']],
                                                target_reindex_map[row['choice4']]]],
                         'choice_made': [choice_idx],
                         'correct_option': [target_reindex_map[row['object correct']]],
                         'trial_type': [trial_type_reindex_map[row['Task type']]]}
            yield _tensorify_trials(construct, device=self.device)

    def __repr__(self):
        return self.name

    def __init__(self, path_to_csv, dataset_name, dev='cuda', ignore_tasks=[]):
        self.data = pd.read_csv(path_to_csv, index_col='Trial')
        expected_cols = ('Cue', 'object selected', 'object correct', 'Task type', 'choice1', 'choice2', 'choice3', 'choice4')
        if False in [expc in self.data.columns for expc in expected_cols]:
            raise ValueError("All expected Cols must be in data file passed")
        self.trial_type_reindex_map, self.index_map = self.reindex(ignore_tasks)
        self.num_cues = [len(self.index_map[key]['cues'] for key in self.trial_type_reindex_map.keys())]
        self.num_targets = [len(self.index_map[key]['targets'] for key in self.trial_type_reindex_map.keys())]
        self.num_trial_types = len(self.trial_type_reindex_map)
        self.device = torch.device(dev)
        self.name = str(dataset_name)


if __name__ == '__main__':
    loader = MTurk1BehaviorData(path_to_csv='./data_files/jeeves_probe_no_high_guass.csv')
    for mt1trial in loader:
        print('cue', mt1trial.cue_idx, 'options', mt1trial.choice_options)


