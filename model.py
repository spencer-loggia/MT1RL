import torch


class MT1QL:
    def __init__(self, num_cues, num_targets, num_trial_types):
        """
        :param num_cues: number of cue options
        :param num_targets: number of target options
        """

        self.q_init = torch.nn.Parameter(torch.normal(size=(num_cues, num_targets),
                                                      mean=(1 / num_targets),
                                                      std=(1 / num_targets) * .2))
        self.lrs = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=.1, std=.02))
        self.temps = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=1, std=.2))
        self.softmax = torch.nn.Softmax()
        self.optim = torch.optim.Adam(lr=.001, params=[self.lrs] + [self.temps] + [self.q_init])

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
                num_options = trial_batch['choice_options'].shape[1]
                batch_size = len(trial_batch['cue_idx'])
                print("batch", count, "of size", batch_size)
                lr = self.lrs[trial_batch['trial_type']].clone().squeeze()  # size batch
                temp = self.temps[trial_batch['trial_type']].clone()
                option_exp = Q[trial_batch['cue_idx'], trial_batch['choice_options']].clone()
                choice_probs = self.softmax(temp * option_exp)
                is_choice = torch.eq(trial_batch['choice_made'], trial_batch['choice_options'])
                c_prob = choice_probs[is_choice].clone()
                likelihood = torch.mean(c_prob)

                reward = torch.eq(trial_batch['correct_option'], trial_batch['choice_made']).squeeze().float()
                max_exp_reward = torch.max(option_exp, dim=1)[0]
                current_value = Q[trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()].clone()
                print(current_value.shape)
                Q.data[trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()] = (1 - lr) * current_value + lr * (reward + max_exp_reward)
                count += 1
                (-1 * likelihood).backward() # we're minimizing
                lepoch += likelihood.detach().item()
                self.optim.step()

                epoch_loss.append(lepoch / count)
                print('liklihood', epoch_loss[-1], '\nlearning rates: ', self.lrs, '**********')

        return epoch_loss


if __name__ == '__main__':
    from dataloader import MTurk1BehaviorData
    jeeves_probe_no_high_gauss_data = MTurk1BehaviorData('./data_files/jeeves_probe_no_high_guass.csv')
    jeeves_probe = MT1QL(num_cues=jeeves_probe_no_high_gauss_data.num_cues,
                         num_targets=jeeves_probe_no_high_gauss_data.num_targets,
                         num_trial_types=jeeves_probe_no_high_gauss_data.num_trial_types)
    jeeves_probe.fit(jeeves_probe_no_high_gauss_data, epochs=1000, dev='cpu')
