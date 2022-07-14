import os.path

import torch
import pickle

class MT1QL:
    def __init__(self, num_cues, num_targets, num_trial_types, dev='cuda'):
        """
        :param num_cues: number of cue options
        :param num_targets: number of target options
        """
        self.device = torch.device(dev)
        self.q_init = torch.nn.Parameter(torch.normal(size=(num_cues, num_targets),
                                                      mean=(1 / num_targets),
                                                      std=(1 / num_targets) * .2,
                                                      device=self.device))
        self.lrs = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=.01, std=.002, device=self.device))
        self.temps = torch.nn.Parameter(torch.normal(size=(num_trial_types,), mean=2, std=.2, device=self.device))
        self.softmax = torch.nn.Softmax()
        self.optim = torch.optim.Adam(lr=.01, params=[self.lrs] + [self.temps] + [self.q_init])

    def fit(self, trial_data, epochs=1000, snap_out_dir='./'):
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
            lepoch = torch.Tensor([0.]).to(self.device)
            for trial_batch in trial_data.get_natural_batch():
                lr = self.lrs[trial_batch['trial_type']].clone().squeeze() # size batch
                temp = self.temps[trial_batch['trial_type']].clone()
                option_exp = Q[trial_batch['cue_idx'], trial_batch['choice_options']].clone()
                choice_probs = self.softmax(temp * option_exp)
                is_choice = torch.eq(trial_batch['choice_made'], trial_batch['choice_options'])
                c_prob = choice_probs[is_choice].clone()
                likelihood = torch.mean(c_prob)
                reward = torch.eq(trial_batch['correct_option'], trial_batch['choice_made']).squeeze().float()
                current_value = Q[trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()].clone()
                Q[trial_batch['cue_idx'].squeeze(), trial_batch['choice_made'].squeeze()] = current_value + lr * (reward - current_value)
                count += 1
                lepoch = lepoch + likelihood
            print('liklihood', lepoch / count,
                  '\nlearning rates', self.lrs,
                  '\ntemperatures', self.temps,
                  '\n**********')
            epoch_loss.append(lepoch.detach().item() / count)
            (lepoch * -1).backward()
            self.optim.step()
            if (epoch % 10) == 0:
                with open(os.path.join(snap_out_dir, "snapshot" + str(epoch) + ".pkl"), 'wb') as f:
                    pickle.dump(self, f)
            if (sum(epoch_loss) / len(epoch_loss)) > .95:
                break

        return epoch_loss


if __name__ == '__main__':
    from dataloader import MTurk1BehaviorData
    import sys
    jeeves_probe_no_high_gauss_data = MTurk1BehaviorData(sys.argv[1])
    jeeves_probe = MT1QL(num_cues=jeeves_probe_no_high_gauss_data.num_cues,
                         num_targets=jeeves_probe_no_high_gauss_data.num_targets,
                         num_trial_types=jeeves_probe_no_high_gauss_data.num_trial_types)
    jeeves_probe.fit(jeeves_probe_no_high_gauss_data, epochs=1000, snap_out_dir=sys.argv[2])
