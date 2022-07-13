import pandas as pd
from collections import namedtuple


class MTurk1BehaviorData:

    """
    dataloader to run through mturk1 data
    """
    def reindex(self):
        unique_cues = self.data['Cue'].unique()
        unique_targets = self.data['object correct'].unique()
        cue_reindex = {cue: i for i, cue in enumerate(sorted(unique_cues))}
        target_reindex = {target: i for i, target in enumerate(sorted(unique_targets))}
        return cue_reindex, target_reindex

    def __init__(self, path_to_csv):
        self.data = pd.read_csv(path_to_csv, index_col='Trial')
        expected_cols = ('Cue', 'object selected', 'object correct', 'choice1', 'choice2', 'choice3', 'choice4')
        if False in [expc in self.data.columns for expc in expected_cols]:
            raise ValueError("All expected Cols must be in data file passed")
        self.cue_reindex_map, self.target_reindex_map = self.reindex()
        self.num_cues = len(self.cue_reindex_map)
        self.num_targets = len(self.target_reindex_map)

    def __getitem__(self, item):
        construct = namedtuple('Trial', ['cue_idx', 'choice_options', 'choice_made', 'correct_option'])
        row = self.data.iloc[item]
        trial = construct(self.cue_reindex_map[row['Cue']],
                          (self.target_reindex_map[row['choice1']],
                           self.target_reindex_map[row['choice2']],
                           self.target_reindex_map[row['choice3']],
                           self.target_reindex_map[row['choice4']]),
                          self.target_reindex_map[row['object selected']],
                          self.target_reindex_map[row['object correct']])
        return trial

    def __iter__(self):
        for i in range(len(self.data)):
            trial = self[i]
            yield trial


if __name__ == '__main__':
    loader = MTurk1BehaviorData(path_to_csv='./data_files/jeeves_probe_no_high_guass.csv')
    for mt1trial in loader:
        print('cue', mt1trial.cue_idx, 'options', mt1trial.choice_options)


