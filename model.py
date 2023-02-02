import copy
import os.path
from typing import Union

import torch
import numpy as np
import pickle
import sys


class MT1QL:
    def __init__(self, num_cues, num_targets, num_trial_types, save_dir, unique_lrs=False, unique_initial=False, dev='cuda'):
        """
        :param num_cues: number of cue options
        :param num_targets: number of target options
        """
        self.trial_types = num_trial_types
        self.device = torch.device(dev)
        self.num_cues = num_cues
        self.num_targets = num_targets
        if unique_initial:
            self.q_init = torch.nn.Parameter(torch.normal(size=(num_trial_types, num_cues, num_targets),
                                                          mean=0.,
                                                          std=.05,
                                                          device=self.device))
            self.in_q_init = None
        else:
            self.q_init = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=0, std=.05, device=self.device))
            self.in_q_init = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=0, std=.05, device=self.device))
        if unique_lrs:
            self.lrs = torch.nn.Parameter(
                torch.normal(size=(num_trial_types, num_cues), mean=-1., std=.05, device=self.device))
            self.temps = torch.nn.Parameter(
                torch.normal(size=(num_trial_types, num_cues), mean=0, std=.05, device=self.device))
        else:
            self.lrs = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=-1, std=.05, device=self.device))
            self.temps = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=0, std=.05, device=self.device))
        self.unique_lrs = unique_lrs
        self.unique_initial = unique_initial
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.optim = torch.optim.Adam(lr=.01, params=[self.lrs] + [self.temps] + [self.q_init])
        self.save_dir = save_dir

    def to(self, device):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.q_init = self.q_init.detach().to(self.device).clone()
        self.lrs = self.lrs.detach().to(self.device).clone()
        self.temps = self.temps.detach().to(self.device).clone()
        return self

    def _initialize_q(self):
        if self.unique_initial:
            # an individual initial value for every cue - target pair
            Q = self.q_init.clone()
            Q = torch.sigmoid(Q).clone()
        else:
            # a single initial value for the correct diagonal on each trial type.
            Q = torch.zeros((self.trial_types, self.num_cues, self.num_targets), device=self.device)
            correct_vals = self.q_init.clone()
            correct_vals = torch.sigmoid(correct_vals).clone()
            incorrect_vals = self.in_q_init.clone()
            incorrect_vals = torch.sigmoid(incorrect_vals).clone().reshape((-1, 1, 1))
            Q = Q + incorrect_vals
            correct_ind = torch.tile(torch.eye(self.num_cues, self.num_targets, device=self.device, dtype=bool),
                                     (self.trial_types, 1, 1))
            Q[correct_ind] = torch.tile(correct_vals[:, None], (1, min(self.num_cues, self.num_targets))).flatten()
        return Q

    def learn_loop(self, trial_data, batch=True, combine_likelihoods=True, sameple_q=None):
        Q = self._initialize_q()
        count = 0
        q_sample = []
        if combine_likelihoods:
            likelihoods = torch.Tensor([0.]).to(self.device)
        else:
            likelihoods = [list() for _ in range(self.trial_types)]
        if batch:
            iter = trial_data.get_natural_batch
        else:
            iter = trial_data.__iter__
        for trial_batch in iter():
            if self.unique_lrs:
                lr = torch.sigmoid(self.lrs[trial_batch['trial_type'], trial_batch['cue_idx']]).clone()  # size batch
                temp = torch.sigmoid(self.temps[trial_batch['trial_type'], trial_batch['cue_idx']]).clone()
            else:
                lr = torch.sigmoid(self.lrs[trial_batch['trial_type']]).clone() / 2
                temp = torch.abs(self.temps[trial_batch['trial_type']]).clone()
            option_exp = Q[trial_batch['trial_type'], trial_batch['cue_idx'], trial_batch['choice_options']].clone()
            choice_probs = self.softmax(temp * option_exp)
            is_choice = torch.eq(trial_batch['choice_made'], trial_batch['choice_options'])
            c_prob = choice_probs[is_choice].clone()
            likelihood = torch.sum(c_prob)
            if combine_likelihoods:
                likelihoods = likelihoods + likelihood
            else:
                likelihoods[trial_batch['trial_type']].append(likelihood.detach().cpu().item())
            reward = torch.eq(trial_batch['correct_option'], trial_batch['choice_made']).float()
            current_value = Q[trial_batch['trial_type'], trial_batch['cue_idx'], trial_batch['choice_made']].clone()
            Q[trial_batch['trial_type'], trial_batch['cue_idx'], trial_batch['choice_made']] = current_value + lr * (reward - current_value)
            if sameple_q is not None and (count % sameple_q) == 0:
                q_sample.append(Q.cpu().detach().numpy().reshape((-1, self.num_cues, self.num_targets)))
            count += 1
        if sameple_q is not None:
            return likelihoods, q_sample
        return likelihoods

    def free_behavior(self, trial_data, sample_q=100):
        Q = self._initialize_q()
        count = 0
        q_sample = []
        rewarded = [list() for _ in range(self.trial_types)]
        iter = trial_data.__iter__
        for trial_batch in iter():
            lr = torch.sigmoid(self.lrs[trial_batch['trial_type']]).clone() / 2  # size batch
            temp = torch.abs(self.temps[trial_batch['trial_type']]).clone()
            option_exp = Q[trial_batch['trial_type'], trial_batch['cue_idx'], trial_batch['choice_options']].clone()
            choice_probs = self.softmax(temp * option_exp)
            np_probs = choice_probs.detach().cpu().numpy().squeeze()
            is_choice = np.random.choice(np.arange(len(np_probs)), p=np_probs)
            choice_made = trial_batch['choice_options'][:, is_choice]
            reward = torch.eq(trial_batch['correct_option'], choice_made).float()
            rewarded[trial_batch['trial_type']].append(reward.detach().cpu().item())
            current_value = Q[trial_batch['trial_type'], trial_batch['cue_idx'], choice_made].clone()
            Q[trial_batch['trial_type'], trial_batch['cue_idx'], choice_made] = current_value + lr * (
                    reward - current_value)
            if sample_q is not None and (count % sample_q) == 0:
                q_sample.append(Q.cpu().detach().numpy().reshape((-1, 14, 14)))
            count += 1
        return rewarded, q_sample

    def predict(self, trial_data, real_behavior=True):
        """
        make choices as the model would seperated by task type. Return model correct / incorrect over trials, probability
        of monkeys choice over trials, and Q matrix over time.
        :return:
        """
        with torch.no_grad():
            if real_behavior:
                res = self.learn_loop(trial_data, batch=False, combine_likelihoods=False, sameple_q=100)
            else:
                res = self.free_behavior(trial_data, sample_q=100)
        return res

    def fit(self, trial_data, epochs=1000):
        """
        :param trial_data: see predict
        :param epochs: number of epochs to run
        :param dev: device to optmize on
        :return:
        """
        print("starting model...", trial_data.name)
        print("Trial types...", self.trial_types)
        print("Cues per trial...", self.num_cues)
        print("Options per cue...", self.num_targets)
        print("LR tensor shape...", self.lrs.shape)
        print("Temperature shape...", self.temps.shape)
        epoch_loss = []
        for epoch in range(epochs):
            print("**********\n", str(trial_data), "EPOCH", epoch)
            self.optim.zero_grad()
            lepoch = self.learn_loop(trial_data, batch=True)
            print('likelihood', lepoch, 'init:', torch.sigmoid(self.q_init.detach().cpu()), 'lr:', torch.sigmoid(self.lrs.detach().cpu()) / 2,
                  'temp:', torch.abs(self.temps.detach().cpu()), 'wrong init', torch.sigmoid(self.in_q_init.detach().cpu()), '\n**********')
            epoch_loss.append(lepoch.cpu().detach().item())
            (lepoch * -1).backward()
            self.optim.step()
            if (epoch % 10) == 0 or epoch == (epochs - 1):
                if epoch == (epochs - 1) or \
                        (len(epoch_loss) > 5 and abs(epoch_loss[-1]) - abs(epoch_loss[-2]) < .2 and
                         abs(epoch_loss[-2]) - abs(epoch_loss[-3]) < .2):
                    with open(os.path.join(self.save_dir, "Asnapshot_final_" + str(epoch) + ".pkl"), 'wb') as f:
                        pickle.dump(self, f)
                    return epoch_loss
                with open(os.path.join(self.save_dir, "snapshot" + str(epoch) + ".pkl"), 'wb') as f:
                    pickle.dump(self, f)
            sys.stdout.flush()
        return epoch_loss


if __name__ == '__main__':
    from dataloader import MTurk1BehaviorData
    import sys

    jeeves_probe_no_high_gauss_data = MTurk1BehaviorData(sys.argv[1])
    jeeves_probe = MT1QL(num_cues=jeeves_probe_no_high_gauss_data.num_cues,
                         num_targets=jeeves_probe_no_high_gauss_data.num_targets,
                         num_trial_types=jeeves_probe_no_high_gauss_data.num_trial_types)
    jeeves_probe.fit(jeeves_probe_no_high_gauss_data, epochs=1000)
