##2D Squash Trainer
import pygame
import random
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from squash import Squash

##Parameters
input_dim = 5
h1 = 20
h2 = 10
output_dim = 2
n_episodes = 100000
batch_size = 128
gamma = 0.99
epsilon = 0.9
eps_end = 0.05
eps_decay = 1/10000
target_update = 10
keys = ['left', 'right']
scores = []

##Create DQN
class DQN(nn.Module):
    
    def __init__(self, input_dim, h1, h2, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, output_dim)
        
        #Try out different activation functions for learning
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


##Create Utilities
class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, replay_tuple):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = replay_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            actions = policy_net(state)
            action_index = torch.argmax(actions).item()
            return action_index

    else:
        index = random.randint(0, 1)
        return index

def plot_scores():
    plt.figure(1)
    plt.clf()
    scores_t = torch.tensor(scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    plt.pause(0.001)


def optimize_model():
    if memory.__len__() < batch_size:
        return

    batch_sample = memory.sample(batch_size)
    
    state_batch = torch.cat(tuple(tup[0] for tup in batch_sample))
    action_batch = torch.cat(tuple(torch.tensor([tup[1]]) for tup in batch_sample))
    reward_batch = torch.cat(tuple(tup[3] for tup in batch_sample))

    state_batch = state_batch.view(batch_size, 5)

    ##instead of gather:
    state_values = policy_net(state_batch)
    state_action_values = torch.zeros(batch_size)
    for i in range(batch_size):
        state_action_values[i] = state_values[i][action_batch[i]]

    next_state_values = torch.zeros(batch_size, dtype=torch.float32)
    for i in range(batch_size):
        if batch_sample[i][2] is not None:
            next_states = target_net(batch_sample[i][2])
            action = torch.max(next_states).detach()
            next_state_values[i] = action

    next_state_values = next_state_values.type(torch.FloatTensor)
    reward_batch = reward_batch.type(torch.FloatTensor)
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    #Optimize
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()    


##Set up game
squash = Squash()

##Set up networks, optimizer and memory
target_net = DQN(input_dim, h1, h2, output_dim)
policy_net = DQN(input_dim, h1, h2, output_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
plt.ion()

for i_episode in range(n_episodes):
    
    squash.init_game()
    
    while squash.gameExit is False:
        optimizer.zero_grad()
        current_state = squash.get_state()
        action = select_action(current_state)
        key = keys[action]
        squash.game_step(key)

        reward = torch.tensor([squash.get_reward()])
        if reward.item() == -10:
            new_state = None
        else:
            new_state = squash.get_state()

        squash.set_reward(0)
        
        memory.store((current_state, action, new_state, reward))

        optimize_model()

        if squash.gameExit is True:
            scores.append(squash.score)
            #plot_scores()
            if epsilon > eps_end:
                epsilon -= eps_decay

    #print("Episode " + str(i_episode) + " Done.")

    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())


print("Training Complete")
torch.save(policy_net.state_dict(), 'model_after_' + str(n_episodes) + '_episodes.pt')
