# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:11:02 2021

@author: MYZ
"""
import numpy as np

class WOA:

    def __init__(self, n_agents, max_iter, lower_b, upper_b, dim, bench_f):
        # init args
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.lower_b = lower_b
        self.upper_b = upper_b
        self.dim = dim
        self.bench_f = bench_f
        # init problem
        self.leader_pos = np.zeros(dim)
        self.leader_score = np.inf
        self.positions = self.initialize_pos(n_agents, dim, upper_b, lower_b)


    def initialize_pos(self, n_agents, dim, upper_b, lower_b):
        n_boundaries = len(upper_b) if isinstance(upper_b, list) else 1
        if n_boundaries == 1:
            positions = np.random.rand(n_agents, dim) * (upper_b - lower_b) + lower_b
        else:
            positions = np.zeros([n_boundaries, dim])
            for i in range(dim):
                positions[:,i] = np.random.rand(n_agents, dim) * (upper_b[i] - lower_b[i]) + lower_b[i]
        return positions

    def forward(self):
        t = 0
        conv_curve = np.zeros(self.max_iter)
        leader_p = np.zeros((self.max_iter, self.dim))

        while t < self.max_iter:
            fitness, self.positions, self.leader_score, self.leader_pos = self.get_fitness(self.positions, self.leader_score, self.leader_pos)
            a_1 =  2 * np.cos((t * np.pi) / (2 * self.max_iter))
            a_2 = -1 + t * (-1 / self.max_iter)
            self.positions = self.update_search_pos(self.positions, self.leader_pos, a_1, a_2)
            conv_curve[t] = self.leader_score
            leader_p[t] = self.leader_pos
            t += 1

        return self.leader_score, self.positions, conv_curve, leader_p

    def get_fitness(self, positions, leader_score, leader_pos): 

        for i in range(positions.shape[0]):       #positions的第一层元素数量
            # adjust agents surpassing bounds
            upper_flag = positions[i,:] > self.upper_b
            lower_flag = positions[i,:] < self.lower_b
            positions[i,:] = positions[i,:] * ((upper_flag + lower_flag) < 1) + self.upper_b * upper_flag + self.lower_b * lower_flag
            # objective function
            fitness = self.bench_f.get_fitness(positions[i,:])
            # update leader
            if fitness < leader_score: # change to > if maximizing
                leader_score = fitness
                leader_pos = positions[i,:]
                

        return fitness, positions, leader_score, leader_pos 

    def update_search_pos(self, positions_, leader_pos_, a_1, a_2):
        positions = positions_.copy()
        leader_pos = leader_pos_.copy()
        
        for i in range(positions.shape[0]):
            r_1 = np.random.rand()
            r_2 = np.random.rand()
            A = 2 * a_1 * r_1 - a_1 # Eq. (2.3)
            C = 2 * r_2             # Eq. (2.4)
            b = 1 
            l = (a_2 - 1) * np.random.rand() + 1
            p = np.random.rand()    # p in Eq. (2.6)
            
            for j in range(positions.shape[1]):
                t = 0
                while t < self.max_iter: 
                    w =  np.sin(((t * np.pi) / (2 * self.max_iter)) + np.pi) + 1
                    if p < 0.5:
                        if np.abs(A) >= 1:
                            rand_leader_idx = int(np.floor(self.n_agents * np.random.rand()))
                            x_rand = positions[rand_leader_idx, :]
                            d_x_rand = np.abs(C * x_rand[j] - positions[i, j])     # Eq. (2.7)
                            positions[i, j] = x_rand[j] - A * d_x_rand
                        else:
                            d_leader = np.abs(C * leader_pos[j] - positions[i, j]) # Eq. (2.1)
                            positions[i, j] =  leader_pos[j] - w * A * d_leader 
                    else:
                        dist_to_leader = np.abs(leader_pos[j] - positions[i, j])   # Eq. (2.5)
                        positions[i, j] = w * dist_to_leader *  np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos[j]                   
                    t += 1
        return positions