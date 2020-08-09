from structure2vec import QhatNetwork, ReplayMemory
from mcts import execute_episode
from trainer import Trainer
import tsp_with_ortools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random

use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def compute_reward(permutation, distance_matrix):
    print(permutation)
    current_node = 1
    my_index = 0
    total_dist = 0
    while(current_node < 19):
        next_index = np.where(permutation == current_node + 1)[0]
        total_dist += distance_matrix[my_index, next_index]
        current_node += 1
        my_index = next_index
    next_index = np.where(permutation == 0)[0]
    total_dist += distance_matrix[my_index, next_index]
    total_dist += distance_matrix[next_index, 0]
    return total_dist

class MCTSEnv_Shell(object):
    def next_state(self, state, action):
        new_state = np.copy(state)
        original_action = action
        if(new_state[0, action] != 0):
            for i in range(len(new_state[0, :])):
                if(new_state[0, i] == 0):
                    action = i
                    continue
            if(action == original_action):
                return new_state
        new_state[0, action] = np.count_nonzero(new_state[0, :]) + 1
        return new_state

    def is_done_state(self, state, step_idx):
        for i in range(len(state[0, :])):
            if(state[0, i] == 0):
                return False
        return True

    def initial_state(self):
        raise NotImplementedError

    def get_obs_for_states(self, states):
        return np.array(states)

    def get_return(self, state, step_idx):
        previous_node = 0
        cum_distance = 0
        missed_one = False
        i = 1
        while(i < 20):
            next_index = i+1
            next_node_find = np.where(state[0, :] == next_index)[0]
            if(len(next_node_find) == 0):
                missed_one = True
                break;
            next_node = next_node_find[0]
            cum_distance += state[1:, :][previous_node, next_node]
            previous_node = next_node
            i+=1
        cum_distance += state[1:, :][0, next_node]
        if(missed_one):
            return 0
        return -cum_distance

class MCTSTrainEnv(MCTSEnv_Shell):
    def __init__(self, solve=False):
        self.seq = np.random.randint(100, size=(20, 2))/100
        self.distance_matrix = squareform(pdist(self.seq))
        self.n_actions = 20
        if(solve):
            solver = tsp_with_ortools.Solver(20)
            route, distance = solver.run(self.distance_matrix)
            print("Optimal distance: {0}".format(distance))

    def initial_state(self):
        state = np.zeros((21, 20), dtype = np.float32)
        state[0, 0] = 1
        state[1:, :] = self.distance_matrix
        return state

class MCTSTestEnv(MCTSEnv_Shell):
    def __init__(self, folder):
        self.folder = folder
        self.file_index = 0
        self.file_names = []
        self.n_actions = 20
        with open('%s/paths.txt' % self.folder, 'r') as f:
            for line in f:
                fname = "%s/%s" % (self.folder, line.split('/')[-1].strip())
                self.file_names.append(fname)

    def initial_state(self):
        fname = self.file_names[self.file_index]
        coors = []
        n_nodes = -1
        in_sec = False
        with open(fname, 'r') as f_tsp:
            for l in f_tsp:
                if 'DIMENSION' in l:
                    n_nodes = int(l.split(' ')[-1].strip())
                if in_sec:
                    id, x, y = [int(w.strip()) for w in l.split(' ')]
                    coors.append([float(x) / 1000000.0, float(y) / 1000000.0])
                elif 'NODE_COORD_SECTION' in l:
                    in_sec = True
        if(n_nodes < 20):
            for i in range(20 - n_nodes):
                coors.append(coors[0])
        or_sequence = 100.0 * np.array(coors)
        self.distance_matrix = squareform(pdist(or_sequence))
        self.current_vertices = np.zeros(20, dtype=np.double)
        state = np.zeros((21, 20), dtype=np.float32)
        state[0, 0] = 1
        state[1:, :] = self.distance_matrix
        self.file_index += 1
        return state

class TSP_MCTS_Network(nn.Module):
  def __init__(self, num_vertices):
    nn.Module.__init__(self)
    self.num_vertices = num_vertices
    self.q_network = QhatNetwork(self.num_vertices)
    self.value_dense = nn.Linear(self.num_vertices, 1)

  def forward(self, state):
    q_values = self.q_network(state[:, 0, :], state[:, 1:, :])
    temp_policy = F.softmax(q_values, dim=0) + 0.01
    for i in range(state.size(0)):
        visited_vertices = state[i, 0, :]
        for v in range(len(visited_vertices)):
            if(visited_vertices[v] != 0):
                temp_policy[i, v] = 0.0
        if(temp_policy[i, :].sum() == 0.0):
            temp_policy[i, 0] = 1.0
    policy = torch.transpose(torch.div(torch.transpose(temp_policy, 0, 1),
                                       temp_policy.sum(dim=1)), 0, 1)
    value = self.value_dense(q_values).view(-1)
    return q_values, policy, value

  def step(self, state):
    _, p, v = self.forward(torch.from_numpy(state))
    return p.detach().numpy(), v.detach().numpy()

class RankedReward:
    def __init__(self, capacity, percentile):
        self.capacity = capacity
        self.percentile = percentile
        self.memory = []

    def add_reward(self, reward):
        self.memory.append(reward)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def rank(self, reward):
        r_alpha = np.percentile(self.memory, self.percentile)
        if(reward > r_alpha):
            return 1
        if(reward == r_alpha):
            return 2*random.randint(0, 1) - 1
        return -1

if __name__ == "__main__":
    n_actions = 20
    n_obs = 20

    value_losses = []
    policy_losses = []

    trainer = Trainer(lambda: TSP_MCTS_Network(n_actions), 0.01)
    network = trainer.step_model

    memory = ReplayMemory(500)
    ranker = RankedReward(20, 75)

    test_rewards = []

    for i in range(10000):
        TSPEnv = MCTSTrainEnv()
        obs, pis, returns, total_reward, done_state = execute_episode(network, 16, TSPEnv)
        ranker.add_reward(total_reward)

        actual_reward = compute_reward(obs[-1][0, :], obs[-1][1:, :])
        print("Episode {1} Total Reward: {0}".format(-total_reward, i))
        print("My Route: {0}".format(obs[-1][0, :]))

        for m in range(len(obs)):
            memory.push((obs[m], pis[m], total_reward))

        if(len(memory) < 100):
            print("Skipping Training, Only {0} Samples".format(len(memory)))
            continue

        print("Perfoming Updates")

        transitions = memory.sample(100)
        batch_state, batch_pis, batch_return = zip(*transitions)
        batch_state = np.array(batch_state)
        batch_action = np.array(batch_pis)
        batch_return = np.array(batch_return)

        vl, pl = trainer.train(batch_state, batch_action, batch_return)
        value_losses.append(vl)
        policy_losses.append(pl)


    TestEnv = MCTSTestEnv("/home/cbostanc/GitProjects/graph_comb_opt/data/tsp2d/test_tsp2d/tsp_min-n=15_max-n=20_num-graph=1000_type=clustered")
    for t in range(1000):
        obs, pis, returns, total_reward, done_state = execute_episode(network, 128, TestEnv)
        test_rewards += [-total_reward]

    plt.plot(test_rewards)
    plt.show()
