import os
from typing import List

import matplotlib.pyplot as plt
import torch
import numpy as np
from model import MT1QL
from dataloader import MTurk1BehaviorData
from torch.multiprocessing import Pool, set_start_method
from scipy.ndimage import median_filter


def _fit_wrapper(model, dataset, epochs):
    model.fit(dataset, epochs)

def _predict_wrapper(model, dataset):
    return model.predict(dataset)


class ExperimentManager:

    def __init__(self, name: str, datasets: List[MTurk1BehaviorData], save_dirs: List[str]):
        self.models = []
        self.datasets = datasets
        self.save_dirs = save_dirs
        self.name = name
        for i, dataset in enumerate(datasets):
            self.models.append(MT1QL(dataset.num_cues, dataset.num_targets, dataset.num_trial_types, save_dirs[i]))

    def __repr__(self):
        return self.name

    def fit(self, epochs=500):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        args = list(zip(self.models, self.datasets, [epochs]*len(self.models)))
        with Pool() as p:
            res = p.starmap(_fit_wrapper, args)

    def plot_learning_curves(self, models_to_plot=None):
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
        if models_to_plot is None:
            models_to_plot = list(range(len(self.models)))
        args = [(self.models[idx], self.datasets[idx]) for idx in models_to_plot]
        with Pool() as p:
            model_learn_curves = p.starmap(_predict_wrapper, args)
        fig, axs = plt.subplots((len(models_to_plot)))
        for i, ax in enumerate(axs):
            for trial_type in model_learn_curves[i]:
                smoothed = median_filter(trial_type, 1000)
                ax.plot(smoothed[1000:55000])
            ax.set_title(str(self.datasets[models_to_plot[i]]))
        return model_learn_curves

    def load(self, handle: List[str]):
        if isinstance(handle, str):
            handle = [handle] * len(self.models)
        import pickle
        self.models = [pickle.load(open(os.path.join(savedir, handle[i]), 'rb')) for i, savedir in enumerate(self.save_dirs)]
