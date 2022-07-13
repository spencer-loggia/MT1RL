from abc import ABC
import os
import pandas as pd
from collections import namedtuple

from torch.utils.data import Dataset


class MTurk1BehaviorData(Dataset, ABC):

    """
    dataloader to run through mturk1 data
    """

    def __init__(self, path_to_csv):
        self.data = pd.read_csv(path_to_csv, index_col='Trial')
        expected_cols = ('Cue', 'object selected', 'object correct', 'choice1', 'choice2', 'choice3', 'choice4')
        if False in [expc in self.data.columns for expc in expected_cols]:
            raise ValueError("All expected Cols must be in data file passed")

    def __getitem__(self, item):
        construct = namedtuple('Trial', ['cue_idx', 'choice_options', 'choice_made', 'correct_option'])
        row = self.data.iloc[item]
        trial = construct(row['Cue'],
                          (row['choice1'], row['choice2'], row['choice3'], row['choice4']),
                          row['object selected'],
                          row['object correct'])
        return trial

    def __iter__(self):
        for i, row in self.data.iterrows():
            construct = namedtuple('Trial', ['cue_idx', 'choice_options', 'choice_made', 'correct_option'])
            mt1trial = construct(row['Cue'],
                              (row['choice1'], row['choice2'], row['choice3'], row['choice4']),
                              row['object selected'],
                              row['object correct'])
            yield mt1trial


if __name__ == '__main__':
    loader = MTurk1BehaviorData(path_to_csv='./data_files/jeeves_probe_no_high_guass.csv')
    for mt1trial in loader:
        print('cue', mt1trial.cue_idx, 'options', mt1trial.choice_options)


