import os
import sys
from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np
from model import MT1QL
from dataloader import MTurk1BehaviorData
from torch.multiprocessing import Pool, set_start_method, get_start_method
from scipy.ndimage import uniform_filter
from sklearn.metrics import confusion_matrix

import pickle

probe_data_map = {"low_gauss_shape_to_color": {"task": (1,),
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
                                            "cues_min_max": (0, 28)}
                  }

train_data_map = {
    "low_guass_colored_shape_to_color": {"task": (1,),
                                         "cues_min_max": (0, 14)},
    "high_guass_colored_shape_to_color": {"task": (1,),
                                          "cues_min_max": (14, 28)},
    "low_guass_colored_shape_to_shape": {"task": (2,),
                                         "cues_min_max": (0, 14)},
    "high_guass_colored_shape_to_shape": {"task": (2,),
                                          "cues_min_max": (14, 28)},
    "achromatic_shape_to_shape": {"task": (3,),
                                  "cues_min_max": (0, 28)},
    "color_to_color": {"task": (4,),
                       "cues_min_max": (0, 28)}
                }


def _fit_wrapper(model, dataset, epochs):
    model.fit(dataset, epochs)


def _predict_wrapper(model, dataset, real_behavior=True):
    return model.predict(dataset, real_behavior=real_behavior)


def _set_mp_env():
    set_start_method('spawn')


class ExperimentManager:

    @classmethod
    def from_trained(cls, name, datasets, model_paths, phase):
        exp = cls(name, [], [], phase)
        exp.datasets = datasets
        for mp in model_paths:
            with open(mp, "rb") as m:
                exp.models.append(pickle.load(m))
            exp.save_dirs.append(os.path.dirname(mp))
        return exp

    def __init__(self, name: str, datasets: List[MTurk1BehaviorData], save_dirs: List[str], phase):
        self.models = []
        if phase == 'train':
            self.is_train = True
            self.task_keys = list(train_data_map.keys())
        elif phase == 'probe':
            self.is_train = False
            self.task_keys = list(probe_data_map.keys())
        else:
            raise ValueError

        self.datasets = datasets
        self.save_dirs = save_dirs
        self.name = name
        self.subject_choice_probs = None
        self.model_free_accuracy = None
        self.subject_real_accuracy = None
        self.subject_Q_estimates = None

        for j, dataset in enumerate(datasets):
            self.models.append(MT1QL(dataset.num_cues, dataset.num_targets, dataset.num_trial_types, save_dirs[j]))

    def __repr__(self):
        return self.name

    def fit(self, epochs=500):
        try:
            _set_mp_env()
        except Exception:
            pass
        args = list(zip(self.models, self.datasets, [epochs] * len(self.models)))
        with Pool() as p:
            res = p.starmap(_fit_wrapper, args)

    def get_subject_accuracy(self):
        self.subject_real_accuracy = []
        for i, datset in enumerate(self.datasets):
            data = datset.data
            self.subject_real_accuracy.append([])
            for idx, task_key in enumerate(self.task_keys):
                task_data = data.loc[data['Task type'] == (idx + 1)]
                correct = task_data['object correct'].to_numpy()
                selected = task_data['object selected'].to_numpy()
                self.subject_real_accuracy[i].append((correct == selected).astype(float))

    def get_subject_choice_probs(self):
        try:
            _set_mp_env()
        except Exception:
            pass
        args = [(self.models[idx], dataset) for idx, dataset in enumerate(self.datasets)]
        with Pool() as p:
            res = p.starmap(_predict_wrapper, args)
        self.subject_choice_probs, self.subject_Q_estimates = list(zip(*res))

    def get_model_accuracy(self):
        try:
            _set_mp_env()
        except Exception:
            pass
        args = [(self.models[idx], dataset, False) for idx, dataset in enumerate(self.datasets)]
        with Pool() as p:
            res = p.starmap(_predict_wrapper, args)
        self.model_free_accuracy, _ = list(zip(*res))

    def plot_learning_curves(self, axs, trials_to_plot=50000, window_size=100, models_to_plot=None, type='subject_probs'):
        """
        :param models_to_plot:
        :return:
        """
        if models_to_plot is None:
            models_to_plot = list(range(len(self.models)))
        if type == 'free_behavior':
            all_learn_curves = self.model_free_accuracy
        elif type == 'subject_probs':
            all_learn_curves = self.subject_choice_probs
        elif type == 'subject_behavior':
            all_learn_curves = self.subject_real_accuracy
        else:
            raise ValueError

        if all_learn_curves is None:
            print("must fit data first.")
            return
        for i, ax in enumerate(axs):
            idx = models_to_plot[i]
            learn_curves = all_learn_curves[idx]
            ax.set_title(str(self.datasets[idx]))
            for j, trial_type in enumerate(learn_curves):
                np_trial = np.array(trial_type)
                smoothed = uniform_filter(np_trial, window_size)
                if i == (len(axs) - 1):
                    ax.plot(smoothed[:min(trials_to_plot, len(smoothed))], label=self.task_keys[j])
                else:
                    ax.plot(smoothed[:min(trials_to_plot, len(smoothed))])
        return axs

    def plot_subject_confusion_matrices(self, axs, trial_start, trial_stop):
        for s, dataset in self.datasets:
            for task, task_name in enumerate(self.task_keys):
                task_data = dataset.data.loc[dataset['Task type'] == task]
                correct = task_data['object correct'].numpy()
                selected = task_data['object selected'].numpy()
                conf = confusion_matrix(correct, selected)
                axs[s, task].imshow(conf)
                axs[s, task].set_title(str(dataset) + ': ' + task_name)
        return axs

    def load(self, handle: List[str]):
        if isinstance(handle, str):
            handle = [handle] * len(self.models)
        import pickle
        self.models = [pickle.load(open(os.path.join(savedir, handle[i]), 'rb')) for i, savedir in
                       enumerate(self.save_dirs)]


if __name__ == '__main__':
    import pickle
    name = sys.argv[1]
    i = 2
    dataset_paths = []
    while True:
        try:
            dataset_paths.append(sys.argv[i])
        except IndexError:
            break
        i += 1
    trials_to_load = [30000, 30000, 60000, 60000, 75000]
    datasets = [MTurk1BehaviorData(dset, os.path.basename(dset.split('.')[0]), trials_to_load=trials_to_load[i]) for i, dset in enumerate(dataset_paths)]
    save_dirs = [os.path.join('models', dset.name) for dset in datasets]
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    runner = ExperimentManager(name, datasets, save_dirs)
    

    trials_to_load = [30000, 30000, 150000, 150000, 150000]
    runner.fit(1250)
    with open(os.path.join('models', name + '.pkl'), 'wb') as f:
        pickle.dump(runner, f)
