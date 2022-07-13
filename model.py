import torch
from torch.utils import data


class MT1QL:
    def __init__(self, num_cues, num_targets):
        """
        :param num_cues: number of cue options
        :param num_targets: number of target options
        """

        self.q_init = torch.nn.Parameter(torch.normal(size=(num_cues, num_targets),
                                                      mean=(1 / num_targets),
                                                      std=(1 / num_targets) * .2))
        self.lrs = [torch.nn.Parameter(torch.normal(size=(1,), mean=.1, std=.02)) for _ in range(num_cues)]
        self.temps = [torch.nn.Parameter(torch.normal(size=(1,), mean=1, std=.2)) for _ in range(num_cues)]
        self.softmax = torch.nn.Softmax()
        self.optim = torch.optim.Adam(lr=.1, params=self.lrs + self.temps + [self.q_init])

    def predict(self, trial_data, grad=False):
        """
        :param trial_data: a dataloader that _get_item_ 's trials with fields:
                        - cue_idx: the index of the cue state
                        - cue_name: human-readable name of cue state
                        - choice_options: tuple of idxs of choice options
                        - choice_made: choice the subject had made
                        - correct_option: idx of correct option
        :param grad: whether to use a gradient
        :return:
        """
        Q = self.q_init.clone()
        count = 0
        loglikelihood = 0
        for trial in trial_data:
            lr = self.lrs[trial.cue_idx]
            temp = self.temps[trial.cue_idx]
            option_exp = Q[trial.cue_idx, trial.choice_options]
            choice_probs = self.softmax(temp * option_exp)
            c_prob = choice_probs(trial.choice_options.index(trial.choice_made))
            loglikelihood = loglikelihood + torch.log(c_prob)

            reward = float(trial.correct_option == trial.choice_made)
            max_exp_reward = torch.max(option_exp)

            Q[trial.cue_idx,
              trial.choice_made] = (1 - lr) * Q[trial.cue_idx,
                                                trial.choice_made] + lr * (reward + max_exp_reward)
            count += 1
        if not grad:
            loglikelihood.detach().clone() / count
        else:
            return loglikelihood / count

    def fit(self, trial_data: data.DataLoader, epochs=1000):
        """
        :param trial_data: see predict
        :param epochs: number of epochs to run
        :return:
        """
        epoch_loss = []
        for epoch in range(epochs):
            print('**********\nEPOCH', epoch)
            self.optim.zero_grad()
            ll = self.predict(trial_data, grad=True)
            epoch_loss.append(ll.detach().item())
            (-1 * ll).backward()  # we're minimizing
            print('loss', epoch_loss[-1], '\nlearning rates: ', self.lrs, '**********')
            self.optim.step()
        return epoch_loss


