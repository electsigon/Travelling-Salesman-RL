import pointer_network
import tsp_with_ortools

import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

BATCH_SIZE = 128
lr=0.001

all_rewards = []

class QNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        nn.Module.__init__(self)
        self.encoder = pointer_network.Encoder(embedding_dim, hidden_size, bidirectional=False)
        self.embedding = nn.Linear(2, embedding_dim, bias=False)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, input_seq, input_lengths):
        embed = self.embedding(input_seq)
        encoder_outputs, encoder_hidden = self.encoder(embed, input_lengths)
        output = F.relu(self.l1(encoder_hidden[0]))
        output1 = F.relu(self.l2(output))
        output2 = self.l3(output1)
        return output2

critic = QNetwork(128, 128)
target_network = QNetwork(128, 128)
policy = pointer_network.PointerNet(2, 128, 128)

#optimizer = optim.Adam(list(critic.parameters()) + list(policy.parameters()), lr)
critic_optimizer = optim.Adam(critic.parameters(), lr)
policy_optimizer = optim.Adam(policy.parameters(), lr)

def solve_instance(dmat):
    solver = tsp_with_ortools.Solver(20)
    route, distance = solver.run(dmat)
    return distance

def compute_reward(sequence, batch_seq):
    total_distance = torch.zeros(batch_seq.size()[0])
    for i in range(batch_seq.size()[0]):
        my_seq = sequence[i]
        my_points = batch_seq[i, :, :]
        current_vertex = my_seq[0]
        for vertex in my_seq[1:]:
            distance = torch.norm(my_points[current_vertex, :] - my_points[vertex, :])
            total_distance[i] += distance
            current_vertex = vertex
        distance = torch.norm(my_points[my_seq[-1], :] - my_points[my_seq[0], :])
        total_distance[i] += distance
        distance_matrix = squareform(pdist(my_points.numpy()))
        opt = solve_instance(distance_matrix)
        all_rewards.append(total_distance[i]/opt)
    print("Average Opt Ratio: {0}".format(np.mean(all_rewards[-BATCH_SIZE:])))
    return total_distance

def learn_step(actor, critic, batch_seq, batch_length, t):
    log_pointer_score, selected_log_scores, argmax_pointer, mask = actor(batch_seq, batch_length)
    log_softmax = torch.sum(selected_log_scores, 1)

    reward = compute_reward(argmax_pointer.detach(), batch_seq)
    critic_score = torch.flatten(target_network(batch_seq, batch_length))

    reward_baseline = reward - critic_score.detach()
    policy_loss = torch.mean(reward_baseline * log_softmax)
    critic_loss = F.mse_loss(torch.flatten(critic(batch_seq, batch_length)), reward)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    print("Episode {2} Policy and Critic Losses: {0}, {1}".format(policy_loss, critic_loss, t))

def gen_random_seq(batch_size, length):
    output = torch.zeros(batch_size, length, 2)
    for i in range(batch_size):
        seq = np.random.randint(100, size=(length, 2))
        pca = PCA(n_components = 2)
        sequence = pca.fit_transform(seq)
        seq = sequence/100.0
        output[i, :, :] = torch.Tensor(seq)
    return output

def train(epoch):
    for t in range(epoch):
        batch_seq = Variable(gen_random_seq(BATCH_SIZE, 20))
        batch_len = Variable(20 * torch.ones(BATCH_SIZE))
        learn_step(policy, critic, batch_seq, batch_len, t)
        if(t % 100 == 0):
            target_network.load_state_dict(critic.state_dict())

if __name__ == "__main__":
    train(5000)

    rewards_t = torch.FloatTensor(all_rewards)
    means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))

    plt.title("Costs during Training of Pointer Network AC")
    plt.ylabel("Total Reward")
    plt.xlabel("Episode")
    plt.plot(all_rewards)
    plt.plot(means.numpy())
    plt.show()
