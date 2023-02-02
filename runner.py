import os
import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from model import MT1QL
from dataloader import MTurk1BehaviorData
from torch.multiprocessing import Pool, set_start_method, get_start_method
import scipy
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from graspologic.embed import AdjacencySpectralEmbed
from neurotools.embed import MDScale

import pickle

color_vals = [[235, 141, 202, 122, 141, 86, 86, 50, 157, 95, 227, 136, 186, 113],
              [139, 84, 166, 102, 186, 114, 188, 115, 163, 102, 133, 82, 166, 102],
              [167, 97, 90, 52, 123, 72, 205, 119, 254, 149, 240, 139, 187, 109]]
color_vals = (np.array(color_vals) / 255).T

color_names = ["LightRed", "DarkRed", "LightYellow", "DarkYellow", "LightGreen", "DarkGreen",
               "LightTurquiose", "DarkTurquiose", "LightBlue", "DarkBlue", "LightPurple", "DarkPurple",
               "LightGray", "DarkGray"]

expected_means_degrees = []

probe_data_map = {"low_gauss_shape_to_color": {"task": (1,),
                                               "cues_min_max": (0, 14)},
                  "focal_low_gauss_shape_to_color": {"task": (2,),
                                                     "cues_min_max": (0, 14)},
                  "high_gauss_shape_to_color": {"task": (1,),
                                                "cues_min_max": (14, 28)},
                  "focal_high_gauss_shape_to_color": {"task": (2,),
                                                      "cues_min_max": (0, 14)},
                  "low_gauss_color_to_shape": {"task": (3,),
                                               "cues_min_max": (0, 14)},
                  "focal_low_gauss_color_to_shape": {"task": (4,),
                                                     "cues_min_max": (0, 14)},
                  "high_gauss_color_to_shape": {"task": (3,),
                                                "cues_min_max": (0, 14)},
                  "focal_high_gauss_color_to_shape": {"task": (4,),
                                                      "cues_min_max": (0, 14)},
                  "achromatic_shape_to_shape": {"task": (6,),
                                                "cues_min_max": (0, 14)}}

train_data_map = {
    "low_guass_colored_shape_to_color": {"task": (1,),
                                         "cues_min_max": (0, 14)},
    "high_guass_colored_shape_to_color": {"task": (1,),
                                          "cues_min_max": (0, 14)},
    "low_guass_colored_shape_to_shape": {"task": (2,),
                                         "cues_min_max": (0, 14)},
    "high_guass_colored_shape_to_shape": {"task": (2,),
                                          "cues_min_max": (0, 14)},
    "achromatic_shape_to_shape": {"task": (3,),
                                  "cues_min_max": (0, 14)},
    "color_to_color_low": {"task": (4,),
                       "cues_min_max": (0, 14)},
    "color_to_color_high": {"task": (5,),
                           "cues_min_max": (0, 14)}
}


def _fit_wrapper(model, dataset, epochs):
    model.fit(dataset, epochs)


def _predict_wrapper(model, dataset, real_behavior=True):
    return model.predict(dataset, real_behavior=real_behavior)


def _set_mp_env():
    set_start_method('spawn')


class ExperimentManager:

    @classmethod
    def from_trained(cls, name, datasets, model_paths, phase, unique_lrs=False, unique_init=False, dev='cpu'):
        exp = cls(name, [], [], phase, unique_lrs=unique_lrs, unique_init=unique_init, dev=dev)
        exp.datasets = datasets
        exp.color_dist = exp.compute_color_distributions()
        for mp in model_paths:
            with open(mp, "rb") as m:
                exp.models.append(pickle.load(m).to(dev))
            exp.save_dirs.append(os.path.dirname(mp))
        return exp

    def __init__(self, name: str, datasets: List[MTurk1BehaviorData], save_dirs: List[str], phase, unique_lrs=False, unique_init=False, dev='cuda'):
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
        self.unique_lrs = unique_lrs
        self.dark_indices = {1, 3, 5, 7, 9, 11, 13}
        self.light_indices = {0, 2, 4, 6, 8, 10, 12}
        self.warm_indices = {0, 1, 2, 3}
        self.cool_indices = {6, 7, 8, 9}
        for j, dataset in enumerate(datasets):
            self.models.append(MT1QL(dataset.num_cues, dataset.num_targets, dataset.num_trial_types, save_dirs[j],
                                     unique_lrs=unique_lrs, unique_initial=unique_init, dev=dev))
        self.color_dist = self.compute_color_distributions()

    def __repr__(self):
        return self.name

    def compute_color_distributions(self, ):
        color_dist = []
        for s, dataset in enumerate(self.datasets):
            color_dist.append({})
            stds = [10, 40]
            means = [0, 0, 60, 60, 120, 120, 180, 180, 240, 240, 300, 300, None, None]
            for std in stds:
                for idx, color in enumerate(color_names):
                    if color not in color_dist[s]:
                        color_dist[s][color] = []
                    color_dist[s][color].append({'std': std,
                                                 'mean': means[idx]})
        return color_dist

    def fit(self, epochs=500, mp=True):
        try:
            _set_mp_env()
        except Exception:
            pass
        args = list(zip(self.models, self.datasets, [epochs] * len(self.models)))
        if mp:
            with Pool() as p:
                res = p.starmap(_fit_wrapper, args)
        else:
            for arg_set in args:
                _fit_wrapper(*arg_set)

    def get_subject_accuracy(self):
        self.subject_real_accuracy = []
        for i, datset in enumerate(self.datasets):
            data = datset.data
            self.subject_real_accuracy.append([])
            for idx, task_key in enumerate(self.task_keys):
                task_data = data.loc[data['Task type'] == idx]
                correct = task_data['object correct'].to_numpy()
                selected = task_data['object selected'].to_numpy()
                self.subject_real_accuracy[i].append((correct == selected).astype(float))

    def get_subject_choice_probs(self, overwrite=False, mp=True):
        if self.subject_choice_probs is None or overwrite:
            try:
                _set_mp_env()
            except Exception:
                pass
            args = [(self.models[idx], dataset) for idx, dataset in enumerate(self.datasets)]
            if mp:
                with Pool() as p:
                    res = p.starmap(_predict_wrapper, args)
            else:
                res = []
                for arg in args:
                    res.append(_predict_wrapper(*arg))
            self.subject_choice_probs, self.subject_Q_estimates = list(zip(*res))

    def get_model_accuracy(self, overwrite=False):
        if self.model_free_accuracy is None or overwrite:
            try:
                _set_mp_env()
            except Exception:
                pass
            args = [(self.models[idx], dataset, False) for idx, dataset in enumerate(self.datasets)]
            with Pool() as p:
                res = p.starmap(_predict_wrapper, args)
            self.model_free_accuracy, _ = list(zip(*res))

    def plot_learning_curves(self, axs, trials_to_plot=50000, window_size=100, models_to_plot=None,
                             type='subject_probs'):
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

        if all_learn_curves is None and type != 'subject_behavior':
            print("must fit data first.")
            return
        if not isinstance(axs, np.ndarray):
            axs = [axs] * len(models_to_plot)
        for i, idx in enumerate(models_to_plot):
            idx = models_to_plot[i]
            learn_curves = all_learn_curves[idx]
            ax = axs[i]
            ax.set_title(str(self.datasets[idx]))
            for j, trial_type in enumerate(learn_curves):
                np_trial = np.array(trial_type)
                smoothed = uniform_filter(np_trial, window_size)
                if i == (len(axs) - 1) and j < len(self.task_keys):
                    ax.plot(smoothed[:min(trials_to_plot, len(smoothed))], label=self.task_keys[j])
                else:
                    ax.plot(smoothed[:min(trials_to_plot, len(smoothed))])
        return axs

    def _get_confusion_matrix(self, task_data, standardize, error_only):
        num_items = len(pd.unique(task_data['Cue']))
        if num_items < 14:
            print("not all cond present")
        correct = task_data['object correct'].to_numpy()
        selected = task_data['object selected'].to_numpy()

        conf = confusion_matrix(correct, selected, labels=list(range(0, num_items)))
        if standardize:
            mod = (np.eye(num_items) * 5.5) + 1
            conf = conf / mod
        if error_only:
            mod = np.logical_not(np.eye(num_items))
            conf = conf * mod
        return conf

    def plot_subject_confusion_matrices(self, axs, trial_start, trial_stop, standardize=True, error_only=False):
        for s, dataset in enumerate(self.datasets):
            data_slice = dataset.data.iloc[trial_start:trial_stop]
            for task, task_name in enumerate(self.task_keys):
                task_data = data_slice.loc[data_slice['Task type'] == task]
                conf = self._get_confusion_matrix(task_data, standardize, error_only)
                axs[task, s].imshow(conf)
                axs[task, s].set_title(str(dataset) + ': ' + task_name)
        return axs

    def plot_color_degree_frequencies(self, axs, trial_start=0, trial_stop=-1, use_selected=False, error_only=False,
                                      task_types=None):
        for s, dataset in enumerate(self.datasets):
            c_data = []
            if task_types is None:
                if self.is_train:
                    task_types = (0, 1, 2, 3)
                else:
                    raise NotImplementedError
            task_data = dataset.data.loc[dataset.data['Task type'].isin(task_types)]
            task_data = task_data.iloc[trial_start:trial_stop]

            if error_only:
                task_data = task_data.loc[task_data["object correct"] != task_data["object selected"]]
            for idx in range(28):
                if idx in (12, 13, 26, 27):
                    # skip grey option
                    colors = np.array([])
                else:
                    if not use_selected:
                        cue_data = task_data.loc[task_data["Cue state"] == idx]
                        colors = cue_data['color degree'].to_numpy()
                    else:
                        selected = task_data.loc[task_data["object selected"] == idx]
                        colors = selected["selected degree"].to_numpy()

                c_data.append(colors)
            axs[s].hist(c_data, bins=360, color=np.tile(color_vals, (2, 1)), stacked=True, alpha=1)
        return axs

    def load(self, handle: List[str]):
        if isinstance(handle, str):
            handle = [handle] * len(self.models)
        import pickle
        self.models = [pickle.load(open(os.path.join(savedir, handle[i]), 'rb')) for i, savedir in
                       enumerate(self.save_dirs)]

    def extract_data(self, trial_start, trial_stop, mode, combine_subjects=True):
        if self.is_train and mode == 'color':
            task_types = (0, 1, 5, 6)
        elif self.is_train and mode == 'shape':
            task_types = (2, 3)
        elif mode == 'color':
            task_types = (0, 2)
        elif mode == 'shape':
            task_types = (4, 6)
        else:
            raise NotImplementedError
        names = [dataset.name for dataset in self.datasets]
        datasets = [dataset.data.iloc[trial_start:trial_stop] for dataset in self.datasets]
        if combine_subjects:
            datasets = [pd.concat(datasets)]
            names = ['all_subjects']
        all_task_data = []
        for s, dataset in enumerate(datasets):
            task_data = dataset.loc[dataset['Task type'].isin(task_types) &
                                    ~dataset['Cue'].isin([12, 13]) &
                                    ~dataset['object selected'].isin([12, 13])]
            all_task_data.append(task_data)
        return all_task_data, names

    def create_similarity_space(self, axs, trial_start, trial_stop, mode='color', embed_dim=2, combine_subjects=True,
                                algorithm='mds', converge_tolerance=.001):
        all_embedded = []
        datasets, names = self.extract_data(trial_start, trial_stop, mode, combine_subjects)
        for s, task_data in enumerate(datasets):
            simmilarity = self._get_confusion_matrix(task_data, standardize=True, error_only=False)
            simmilarity = (simmilarity + simmilarity.T) / 2
            dissim = np.max(simmilarity) - simmilarity
            error_vals = dissim[np.logical_not(np.eye(len(simmilarity), dtype=bool))]
            dissim = (dissim - np.mean(error_vals)) / np.std(error_vals)
            dissim += 5
            dissim[np.eye(len(simmilarity), dtype=bool)] = 0

            if algorithm == 'svd':
                reducer = AdjacencySpectralEmbed(n_components=embed_dim, algorithm='full')
            elif algorithm == 'mds':
                reducer = MDScale(n=12, embed_dims=embed_dim, device='cpu')
            else:
                raise ValueError
            embed = reducer.fit_transform(dissim, max_iter=50000, tol=converge_tolerance)
            embed = [embed]
            embed = [np.array(e) for e in embed]
            right_rms = np.sqrt(np.sum(np.power(dissim, 2), axis=0))
            print(names[s], "distance from others", right_rms.tolist())
            all_embedded.append(embed)
            if not combine_subjects:
                ax = axs[s]
            else:
                ax = axs
            ax[0].set_title(names[s] + " Right Latent")
            #ax[1].set_title(names[s] + " Left Latent")
            maxes = []
            for i in range(len(embed)):
                embed[i][:, 0] -= np.min(embed[i][:, 0])
                embed[i][:, 1] -= np.min(embed[i][:, 1])
                if embed_dim == 3:
                    embed[i][:, 2] -= np.min(embed[i][:, 2])
                maxes.append(np.max(embed[i]))
                pad = maxes[i] * .1
                ax[i].set_xlim(-pad, maxes[i] + pad)
                #ax[i].set_ylim(-pad, maxes[i] + pad)
            if embed_dim == 2:
                ax[0].scatter(embed[0][:, 0], embed[0][:, 1], color=color_vals[:12, :], s=120)
                #ax[1].scatter(embed[1][:, 0], embed[1][:, 1], color=color_vals[:12, :], s=120)
            elif embed_dim == 3:
                ax[0].scatter(embed[0][:, 0], embed[0][:, 1], embed[0][:, 2], color=color_vals[:12, :], s=120)
                #ax[1].scatter(embed[1][:, 0], embed[1][:, 1], embed[1][:, 2], color=color_vals[:12, :], s=120)
        return axs, all_embedded


if __name__ == '__main__':
    import pickle

    train_dataset_paths = ['data_files/fixed_jeevestrain_2afc_og.csv', 'data_files/fixed_woostertrain_2afc_og.csv',
                           'data_files/fixed_jeevestrain_4afc_og.csv', 'data_files/fixed_woostertrain_4afc_og.csv',
                           'data_files/fixed_jocamotrain_4afc_og.csv']
    trials_to_load = [30000, 30000, 60000, 60000, 75000]

    datasets = [MTurk1BehaviorData(dset, os.path.basename(dset.split('.cs')[0]), trials_to_load=trials_to_load[i], dev='cuda') for
                i, dset in enumerate(train_dataset_paths)]
    save_dirs = [os.path.join('models', dset.name) for dset in datasets]
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    runner = ExperimentManager("train", datasets, save_dirs, unique_lrs=False, unique_init=False, phase="train", dev='cuda')
    runner.fit(2000, mp=True)
    with open(os.path.join('models', 'train_class_lrs_class_init.pkl'), 'wb') as f:
        pickle.dump(runner, f)
    del runner

    probe_dataset_paths = ['data_files/fixed_jeevesprobe_2afc_og.csv', 'data_files/fixed_woosterprobe_2afc_og.csv',
                           'data_files/fixed_jeevesprobe_4afc_og.csv', 'data_files/fixed_woosterprobe_4afc_og.csv',
                           'data_files/fixed_jocamoprobe_4afc_og.csv']
    trials_to_load = [40000, 40000, 130000, 130000, 130000]
    datasets = [MTurk1BehaviorData(dset, os.path.basename(dset.split('.cs')[0]), trials_to_load=trials_to_load[i], dev='cuda') for
                i, dset in enumerate(probe_dataset_paths)]
    save_dirs = [os.path.join('models', dset.name) for dset in datasets]
    for save_dir in save_dirs:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    runner = ExperimentManager("probe", datasets, save_dirs, unique_lrs=False, unique_init=False, phase="probe", dev='cuda')
    runner.fit(2000, mp=True)
    with open(os.path.join('models', 'probe_class_lrs_class_init.pkl'), 'wb') as f:
        pickle.dump(runner, f)
