import torch
from parameters import * 

class QValues():
    def get_current(policy_net, state_batch, action_batch):
        return policy_net(state_batch.float()).gather(1, action_batch)

    def get_next(policy_net, target_net, next_state_batch, next_mask, device):
        next_state_values = torch.zeros(target_net.batch_size, device=device)
        if(rl == 'DQN'):
            next_state_values[next_mask] = target_net(next_state_batch.float()).max(1)[0].detach()
        elif(rl == 'DDQN'):
            # Get the best action for the next state using the policy network
            argmax = policy_net(next_state_batch.float()).max(1)[1].unsqueeze(0).t()
            # Get the value of the action selected by the policy, using the target
            next_state_values[next_mask] = target_net(next_state_batch.float()).gather(1, argmax).detach().t().squeeze(0)
        return next_state_values