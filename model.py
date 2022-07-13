import torch
torch.autograd.set_detect_anomaly(True)


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
        self.optim = torch.optim.Adam(lr=.001, params=self.lrs + self.temps + [self.q_init])

    def fit(self, trial_data, epochs=1000, dev='cuda:0'):
        """
        :param trial_data: see predict
        :param epochs: number of epochs to run
        :param dev: device to optmize on
        :return:
        """
        epoch_loss = []
        for epoch in range(epochs):
            print('**********\nEPOCH', epoch)
            self.optim.zero_grad()
            Q = self.q_init.clone()
            count = 0
            lepoch = 0
            for trial_batch in trial_data.get_natural_batch():
                print("batch", count, "of size", trial_batch['batch_size'])
                lr = self.lrs[trial_batch['cue_idx']]  # size batch
                temp = self.temps[trial_batch['cue_idx']]
                option_exp = Q[trial_batch['cue_idx'], trial_batch['choice_options']].clone()
                choice_probs = self.softmax(temp * option_exp)
                c_prob = choice_probs[
                    torch.nonzero(torch.logical_and(trial_batch['choice_made'], trial_batch['choice_options']))].clone()
                likelihood = torch.mean(c_prob)

                reward = torch.logical_and(trial_batch['correct_option'], trial_batch['choice_made']).float()
                max_exp_reward = torch.max(option_exp, dim=1)

                Q.data[trial_batch['cue_idx'], trial_batch['choice_made']] = (1 - lr) * Q[
                    trial_batch['cue_idx'], trial_batch['choice_made']].clone() + lr * (reward + max_exp_reward)
                count += 1
                (-1 * likelihood).backward() # we're minimizing
                lepoch += likelihood.detach().item()
                self.optim.step()

            epoch_loss.append(lepoch / count)
            print('loss', epoch_loss[-1], '\nlearning rates: ', self.lrs, '**********')

        return epoch_loss


