from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from MecOpt import MecEnv
import math
import random

print(torch.__version__)


EPS_START = 0.8
EPS_END = 0.01
EPS_DECAY = 2000
steps_done = 0


class QNet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        hidden1 = 3 * n_outputs
        hidden2 = 2 * n_outputs
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_inputs, hidden1)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden2, n_outputs)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class dqn:
    def __init__(self,
                 n_inputs=1,
                 n_outputs=1,
                 memory_size=1,
                 batch_size=32,
                 learning_rate=1e-3,
                 training_interval=10,
                 epsilon_greedy=0.9,
                 gamma=0.6,
                 ):
        self.memory_low = 1000
        self.state_dim = n_inputs
        self.action_dim = n_outputs
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_interval = training_interval
        self.epsilon_greedy = epsilon_greedy
        self.gamma = gamma
        self.eval_net = QNet(self.state_dim, self.action_dim)
        self.target_net = QNet(self.state_dim, self.action_dim)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, s):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        s = Variable(torch.unsqueeze(torch.Tensor(s), 0))
        if sample > eps_threshold:
            action = torch.max(self.eval_net(s), 1)[1].data[0]
            return action
        else:
            return random.randrange(self.action_dim)

    def store_memory(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net parameter update
        # sample experience
        # data from mini batch
        if self.memory_low <= self.memory_counter < self.memory_size:
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        elif self.memory_counter >= self.memory_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            return
        if self.learn_step_counter % self.training_interval == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample experience
        # data from mini batch
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.state_dim]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.state_dim:self.state_dim + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.state_dim + 1: self.state_dim + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.state_dim:]))
        self.eval_net.eval()
        self.target_net.eval()
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.criterion(q_target, q_eval)  # MSE loss
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
