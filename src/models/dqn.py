import torch
import copy
import numpy as np
from network import ConvNet

class DQNAgent(object):
    def __init__(
        self,
        num_actions,
        state_dim, #?
        in_channels,
        device,
        discount=0.9,
        optimizer="Adam",
        optimizer_parameters={'lr':0.01},
        target_update_frequency=1e4,
        initial_eps = 1,
        end_eps = 0.05,
        eps_decay_period = 25e4,
        eval_eps=0.001
    ) -> None:
        # Set Device
        self.device = device

        self.Q = ConvNet(state_dim, in_channels, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)  # copy target network
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), 
        **optimizer_parameters)

        self.discount = discount

        self.target_update_frequency = target_update_frequency
 
        # epsilon decay
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval else max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        self.current_eps = eps

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0,1) > eps:
            self.Q.eval()
            with torch.no_grad():
                # without batch norm, remove the unsqueeze
                state = torch.FloatTensor(state).reshape(self.state_shape).unsqueeze(0).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)


    def train(self, replay_memory):
        self.Q.train()
        # Sample mininbatch from replay buffer
        state, action, next_state, reward, done = replay_memory.sample()

        # Convert action tensor to int64 data type
        action = action.clone().detach().long()

        # Compute the target Q value
        with torch.no_grad():
            target_Q = reward + (1-done) * self.discount * self.Q_target(next_state).max(1, keepdim=True)[0]

        # Get current Q estimate
        # torch gather just selects action values from Q(state) using the action tensor as an index
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by full copy every X iterations.
        self.iterations += 1
        self.copy_target_update()
    
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('target network updated')
            print('current epsilon', self.current_eps)
            self.Q_target.load_state_dict(self.Q.state_dict())


    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))