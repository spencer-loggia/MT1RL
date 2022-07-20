import copy
import os.path
from typing import Union

import torch
import numpy as np
import pickle
import sys


class MT1QL:
    def __init__(self, num_cues, num_targets, num_trial_types, save_dir, dev='cuda'):
        """
        :param num_cues: number of cue options
        :param num_targets: number of target options
        """
        self.trial_types = num_trial_types
        self.device = torch.device(dev)
        self.q_init = [torch.nn.Parameter(torch.normal(size=(num_cues[i], num_targets[i]),
                                                      mean=(1 / num_targets[i]),
                                                      std=(1 / num_targets[i]) * .2,
                                                      device=self.device)) for i in range(num_trial_types)]
        self.lrs = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=.01, std=.002, device=self.device))
        self.temps = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=2, std=.2, device=self.device))
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.optim = torch.optim.Adam(lr=.01, params=[self.lrs] + [self.temps] + self.q_init)
        self.save_dir = save_dir

    def to(self, device):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.q_init = self.q_init.to(self.device)
        self.lrs = self.lrs.to(self.device)
        self.temps = self.temps.to(self.device)

    def learn_loop(self, trial_data, batch=True, combine_likelihoods=True):
        q_init = [q.clone() for q in self.q_init]
        Q = [self.sigmoid(q).clone() for q in q_init]
        count = 0
        if combine_likelihoods:
            likelihoods = torch.Tensor([0.]).to(self.device)
        else:
            likelihoods = [list() for _ in range(self.trial_types)]
        if batch:
            iter = trial_data.get_natural_batch
        else:
            iter = trial_data.__iter__
        for trial_batch in iter():
            trial_type = trial_batch['trial_type']
            lr = torch.abs(self.lrs[trial_type].clone().squeeze())  # size batch
            temp = torch.abs(self.temps[trial_type].clone())
            option_exp = Q[trial_type][trial_batch['cue_idx'], trial_batch['choice_options']].clone()
            choice_probs = self.softmax(temp * option_exp)
            is_choice = torch.eq(trial_batch['choice_made'], trial_batch['choice_options'])
            c_prob = choice_probs[is_choice].clone()
            likelihood = torch.mean(c_prob)
            if combine_likelihoods:
                likelihoods += likelihood
            else:
                likelihoods[trial_batch['trial_type']].append(likelihood.detach().cpu().item())
            reward = torch.eq(trial_batch['correct_option'], trial_batch['choice_made']).squeeze().float()
            current_value = Q[trial_type][trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()].clone()
            Q[trial_type][trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()] = current_value + lr * (
                        reward - current_value)
            count += 1
        if combine_likelihoods:
            likelihoods = likelihoods / count
        return likelihoods

    def predict(self, trial_data):
        """
        make choices as the model would seperated by task type. Return model correct / incorrect over trials, probability
        of monkeys choice over trials, and Q matrix over time.
        :return:
        """
        og_dev = copy.copy(self.device)
        self.to('cpu')
        with torch.no_grad():
            choice_made_probs = self.learn_loop(trial_data, batch=False, combine_likelihoods=False)
        self.to(og_dev)
        return np.array(choice_made_probs)

    def fit(self, trial_data, epochs=1000):
        """
        :param trial_data: see predict
        :param epochs: number of epochs to run
        :param dev: device to optmize on
        :return:
        """
        epoch_loss = []
        for epoch in range(epochs):
            print("**********\n", str(trial_data), "EPOCH", epoch)
            self.optim.zero_grad()
            lepoch = self.learn_loop(trial_data, batch=True)
            print('liklihood', lepoch,
                  '\nlearning rates', self.lrs,
                  '\ntemperatures', self.temps,
                  '\n**********')
            epoch_loss.append(lepoch.detach().item())
            (lepoch * -1).backward()
            self.optim.step()
            if (epoch % 10) == 0:
                with open(os.path.join(self.save_dir, "snapshot" + str(epoch) + ".pkl"), 'wb') as f:
                    pickle.dump(self, f)
            if (sum(epoch_loss) / len(epoch_loss)) > .95:
                break
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
