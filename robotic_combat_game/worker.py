import copy
import os
import networkx as nx
import random

import imageio
import numpy as np
import torch
from env import Env
from parameter import *


class Worker:
    def __init__(self, meta_agent_id, policy_net_self, policy_net_enemy, q_net, global_step, device='cuda', greedy=False, save_image=False, random_seed=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(k_size=self.k_size, plot=save_image, num_self=N_SELF, random_seed=random_seed)
        self.local_policy_net_self = policy_net_self
        self.local_policy_net_enemy = policy_net_enemy
        self.local_q_net = q_net

        self.current_node_index = 0

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(29):
            self.episode_buffer.append([])

    def get_observations(self, robot_index, side):
        # get observations
        graph = self.env.graph
        if side == 0:
            node_feature = self.env.node_feature_self
        else:
            node_feature = self.env.node_feature_enemy
        
        # normalize observations
        # node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
        
        # transfer to node inputs tensor
        node_feature_inputs = node_feature.reshape((self.env.node_num, -1))
        # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
        node_inputs = node_feature_inputs
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

        # padding the number of node to a given node padding size
        assert self.env.node_num < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - self.env.node_num))
        node_inputs = padding(node_inputs)

        # calculate a mask for padded node
        node_padding_mask = torch.zeros((1, 1, self.env.node_num), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - self.env.node_num), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        enemy_input = []
        if side == 0:
            current_node_index = self.env.self_indexes[robot_index]
            for i in range(N_ENEMY):
                # print('curr:', current_node_index, 'enemy:', self.env.enemy_indexes[i], 'shoot:', self.env.shoot_matrix[current_node_index][self.env.enemy_indexes[i]])
                if self.env.enemy_flags[i] and self.env.shoot_matrix[current_node_index][self.env.enemy_indexes[i]] == 1:
                    enemy_input.append(self.env.enemy_indexes[i])

            while len(enemy_input) < N_ENEMY:
                enemy_input.append(-1)

        elif side == 1:
            current_node_index = self.env.enemy_indexes[robot_index]
            for i in range(N_SELF):
                if self.env.self_flags[i] and self.env.shoot_matrix[current_node_index][self.env.self_indexes[i]] == 1:
                    enemy_input.append(self.env.self_indexes[i])
            
            while len(enemy_input) < N_SELF:
                enemy_input.append(-1)

        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        enemy_input = torch.tensor(enemy_input).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, n)
        enemy_inputs = torch.where(enemy_input == -1, 0, enemy_input)
        enemy_padding_mask = torch.zeros((1, 1, N_ENEMY), dtype=torch.int64).to(self.device)
        one = torch.ones_like(enemy_padding_mask, dtype=torch.int64).to(self.device)
        enemy_padding_mask = torch.where(enemy_input == -1, one, enemy_padding_mask)
        # print('enemy_inputs: ', enemy_inputs.shape, 'enemy_padding_mask: ', enemy_padding_mask.shape)

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        edge_inputs = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors = [n for n in neighbors if n not in self.env.self_indexes + self.env.enemy_indexes]
            if node in neighbors:
                neighbors.remove(node)
            edge_list = [node] + neighbors  # 再将当前节点放在最前面
            edge_inputs.append(edge_list)
        edge_for_select_node = edge_inputs

        adjacent_matrix = self.env.adjacent_matrix
        # print('adj matrix: ', adjacent_matrix)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(-1)

        edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
        edge_inputs = torch.where(edge_input == -1, 0, edge_input)
        
        # calculate a mask for the padded edges (denoted by -1)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
        observations = node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask
        
        return observations, edge_for_select_node
    
    def get_all_observations(self, side):
        # get observations
        graph = self.env.graph
        if side == 0:
            node_feature = self.env.node_feature_self
        else:
            node_feature = self.env.node_feature_enemy
        
        # normalize observations
        # node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
        
        # transfer to node inputs tensor
        node_feature_inputs = node_feature.reshape((self.env.node_num, -1))
        # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
        node_inputs = node_feature_inputs
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

        # padding the number of node to a given node padding size
        assert self.env.node_num < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - self.env.node_num))
        node_inputs = padding(node_inputs)

        # calculate a mask for padded node
        node_padding_mask = torch.zeros((1, 1, self.env.node_num), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - self.env.node_num), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # get the node index of the current robot position
        all_current_indexes = []
        all_enemy_input = []
        all_enemy_padding_mask = []
        if side == 0:
            for j in range(N_SELF):
                enemy_input = []
                current_node_index = self.env.self_indexes[j]
                all_current_indexes.append(current_node_index)

                for i in range(N_ENEMY):
                    if self.env.enemy_flags[i] and self.env.shoot_matrix[current_node_index][self.env.enemy_indexes[i]] == 1:
                        enemy_input.append(self.env.enemy_indexes[i])

                while len(enemy_input) < N_ENEMY:
                    enemy_input.append(-1)
                
                enemy_input = torch.tensor(enemy_input).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, n)
                enemy_inputs = torch.where(enemy_input == -1, 0, enemy_input)
                enemy_padding_mask = torch.zeros((1, 1,N_ENEMY), dtype=torch.int64).to(self.device)
                one = torch.ones_like(enemy_padding_mask, dtype=torch.int64).to(self.device)
                enemy_padding_mask = torch.where(enemy_input == -1, one, enemy_padding_mask)
                all_enemy_input.append(enemy_inputs)
                all_enemy_padding_mask.append(enemy_padding_mask)

        elif side == 1:
            for j in range(N_ENEMY):
                enemy_input = []
                current_node_index = self.env.enemy_indexes[j]
                all_current_indexes.append(current_node_index)

                for i in range(N_SELF):
                    if self.env.self_flags[i] and self.env.shoot_matrix[current_node_index][self.env.self_indexes[i]] == 1:
                        enemy_input.append(self.env.self_indexes[i])
            
                while len(enemy_input) < N_SELF:
                    enemy_input.append(-1)
                
                enemy_input = torch.tensor(enemy_input).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, n)
                enemy_inputs = torch.where(enemy_input == -1, 0, enemy_input)
                enemy_padding_mask = torch.zeros((1, 1, N_ENEMY), dtype=torch.int64).to(self.device)
                one = torch.ones_like(enemy_padding_mask, dtype=torch.int64).to(self.device)
                enemy_padding_mask = torch.where(enemy_input == -1, one, enemy_padding_mask)
                all_enemy_input.append(enemy_inputs)
                all_enemy_padding_mask.append(enemy_padding_mask)

        all_current_index = torch.tensor(all_current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, N_ROBOTS)
        all_enemy_input  = torch.cat(all_enemy_input, dim=-1)   # (1, 1, N_ENEMY*N_SELF)
        all_enemy_padding_mask = torch.cat(all_enemy_padding_mask, dim=-1)   # (1, 1, N_ENEMY*N_SELF)
        # print('all_enemy_input: ', all_enemy_input.shape, 'all_enemy_padding_mask: ', all_enemy_padding_mask.shape)
        
        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        edge_inputs = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors = [n for n in neighbors if n not in self.env.self_indexes + self.env.enemy_indexes]
            if node in neighbors:
                neighbors.remove(node)
            edge_list = [node] + neighbors  # 再将当前节点放在最前面
            edge_inputs.append(edge_list)
        edge_for_select_node = edge_inputs

        adjacent_matrix = self.env.adjacent_matrix
        # print('adj matrix: ', adjacent_matrix)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        # padding edge mask
        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        all_edges_inputs = []
        all_edge_padding_masks = []
        if side == 0:
            for i in range(N_SELF):
                current_index = self.env.self_indexes[i]
                edge = edge_inputs[current_index]
                while len(edge) < self.k_size:
                    edge.append(-1)

                edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
                
                # calculate a mask for the padded edges (denoted by -1)
                edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
                one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
                edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
                edge_input = torch.where(edge_input == -1, 0, edge_input)
                all_edges_inputs.append(edge_input)
                all_edge_padding_masks.append(edge_padding_mask)
                # print('edge_input: ', edge_input.shape, 'edge_padding_mask: ', edge_padding_mask.shape)
        else:
            for i in range(N_ENEMY):
                current_index = self.env.enemy_indexes[i]
                edge = edge_inputs[current_index]
                while len(edge) < self.k_size:
                    edge.append(-1)

                edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

                # calculate a mask for the padded edges (denoted by -1)
                edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
                one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
                edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)
                edge_input = torch.where(edge_input == -1, 0, edge_input)
                all_edges_inputs.append(edge_input)
                all_edge_padding_masks.append(edge_padding_mask)
                
        all_edge_inputs = torch.cat(all_edges_inputs, dim=-1)
        all_edge_padding_mask = torch.cat(all_edge_padding_masks, dim=-1)
        # print('all_edge_inputs: ', all_edge_inputs.shape, 'all_edge_padding_mask: ', all_edge_padding_mask.shape)
        all_observations = node_inputs, all_edge_inputs, all_enemy_input, all_current_index, node_padding_mask, all_edge_padding_mask, edge_mask, all_enemy_padding_mask

        return all_observations, edge_for_select_node
    
    def select_action(self, observations, side):
        node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask = observations
        # print('edge_inputs: ', edge_inputs.shape, 'current_index: ', current_index.shape, 'enemy_inputs: ', enemy_inputs.shape, 'edge_padding_mask: ', edge_padding_mask.shape, 'enemy_padding_mask: ', enemy_padding_mask.shape)
        if side == 0:
            with torch.no_grad():
                logps = self.local_policy_net_self(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
                                                    edge_padding_mask, edge_mask, enemy_padding_mask)
            #     log_ref = self.local_policy_net_enemy(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
            #                                         edge_padding_mask, edge_mask, enemy_padding_mask)
                
            # # 求散度
            # kl_div = torch.nn.functional.kl_div(logp, log_ref, reduction='batchmean')
            # print('kl_div: ', kl_div)
            if self.greedy:
                action_index = torch.argmax(logps, dim=1).long()
            else:
                action_index = torch.multinomial(logps.exp(), 1).long().squeeze(1)
            action_index = action_index.item()
            # print('logp: ', logp, 'action_index: ', action_index)
            # 判断选择的是移动还是攻击
            if action_index < K_SIZE:  # 选择移动
                # 获取目标节点索引
                target_index = edge_inputs[0, 0, action_index].item()
                return 'move', target_index, action_index
            else:  # 选择攻击
                # 将action_index转换为敌人索引
                attack_index = action_index - K_SIZE
                # 获取可攻击的敌人列表
                candidate = []
                for i in range(N_ENEMY):
                    if self.env.enemy_flags[i] and self.env.shoot_matrix[current_index][self.env.enemy_indexes[i]] == 1:
                        candidate.append(i)
                target_index = candidate[attack_index]
                return 'attack', target_index, action_index
        else:
            with torch.no_grad():
                logps = self.local_policy_net_enemy(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
                                                    edge_padding_mask, edge_mask, enemy_padding_mask)
            
            if self.greedy:
                action_index = torch.argmax(logps, dim=1).long()
            else:
                action_index = torch.multinomial(logps.exp(), 1).long().squeeze(1)
            action_index = action_index.item()
            # print('logp: ', logp, 'action_index: ', action_index)
            logp = logps[:, action_index]
            # 判断选择的是移动还是攻击
            if action_index < K_SIZE:  # 选择移动
                # 获取目标节点索引
                target_index = edge_inputs[0, 0, action_index].item()
                return 'move', target_index, action_index, logp
            else:  # 选择攻击
                # 将action_index转换为敌人索引
                attack_index = action_index - K_SIZE
                # 获取可攻击的敌人列表
                candidate = []
                for i in range(N_SELF):
                    if self.env.self_flags[i] and self.env.shoot_matrix[current_index][self.env.self_indexes[i]] == 1:
                        candidate.append(i)
                target_index = candidate[attack_index]
                return 'attack', target_index, action_index, logp
            
    def select_node_random(self, robot_index, observations, edge_for_select_node, side):
        # node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        # if side == 0:
        #     candidate = edge_for_select_node[self.env.self_indexes[robot_index]]
        # elif side == 1:
        #     candidate = edge_for_select_node[self.env.enemy_indexes[robot_index]]
        # candidate = [value for value in candidate if value != -1]
        # next_node_index = random.choice(candidate)
        # action_index = candidate.index(next_node_index)

        # print(candidate, next_node_index, action_index)
        # return next_node_index, action_index
        # ------------------------------------------------------
        # 随机选择动作类型（移动或攻击）
        if side == 0:
            current_index = self.env.self_indexes[robot_index]
            # 获取可攻击的敌人列表
            candidate = []
            for i in range(N_ENEMY):
                if self.env.enemy_flags[i] and self.env.shoot_matrix[current_index][self.env.enemy_indexes[i]] == 1:
                    candidate.append(i)
            # candidate = [i for i, flag in enumerate(self.env.enemy_flags) if flag]
            # 获取可移动的节点列表
            move_candidate = edge_for_select_node[current_index]
            move_candidate = [value for value in move_candidate if value != -1]
        else:
            current_index = self.env.enemy_indexes[robot_index]
            # 获取可攻击的我方单位列表
            candidate = []
            for i in range(N_SELF):
                if self.env.self_flags[i] and self.env.shoot_matrix[current_index][self.env.self_indexes[i]] == 1:
                    candidate.append(i)
            # candidate = [i for i, flag in enumerate(self.env.self_flags) if flag]
            # 获取可移动的节点列表
            move_candidate = edge_for_select_node[current_index]
            move_candidate = [value for value in move_candidate if value != -1]

        # 随机选择动作类型
        if len(candidate) == 0:  # 如果没有可攻击目标，只能移动
            action_type = 'move'
        elif len(move_candidate) == 0:  # 如果无法移动，只能攻击
            action_type = 'attack'
        else:  # 随机选择移动或攻击
            action_type = random.choice(['move', 'attack'])

        if action_type == 'move':
            target_index = random.choice(move_candidate)
            action_index = move_candidate.index(target_index)
        else:
            target_index = random.choice(candidate)
            action_index = K_SIZE + candidate.index(target_index)
        print('action_type:', action_type, 'target_index:', target_index, 'action_index:', action_index)
        return action_type, target_index, action_index
        

    def save_observations(self, self_observations, enemy_observations):
        node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask= self_observations
        self.episode_buffer[0] += copy.deepcopy(node_inputs)
        self.episode_buffer[1] += copy.deepcopy(edge_inputs)
        self.episode_buffer[2] += copy.deepcopy(enemy_inputs)
        self.episode_buffer[3] += copy.deepcopy(current_index)
        self.episode_buffer[4] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[5] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[6] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[7] += copy.deepcopy(enemy_padding_mask).bool()

        node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask= enemy_observations
        self.episode_buffer[8] += copy.deepcopy(node_inputs)
        self.episode_buffer[9] += copy.deepcopy(edge_inputs)
        self.episode_buffer[10] += copy.deepcopy(enemy_inputs)
        self.episode_buffer[11] += copy.deepcopy(current_index)
        self.episode_buffer[12] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[13] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[14] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[15] += copy.deepcopy(enemy_padding_mask).bool()

    def save_action(self, self_action_index, enemy_action_index):
        self_action_index = torch.tensor(self_action_index)
        enemy_action_index = torch.tensor(enemy_action_index)
        self.episode_buffer[16] += copy.deepcopy(self_action_index.unsqueeze(0))
        self.episode_buffer[17] += copy.deepcopy(enemy_action_index.unsqueeze(0))

    def save_logp(self, logp):
        logp = sum(logp)
        logp = torch.tensor(logp)
        self.episode_buffer[18] += copy.deepcopy(logp.unsqueeze(0))

    def save_reward_done(self, reward, done):
        self.episode_buffer[19] += copy.deepcopy(torch.FloatTensor([[[reward]]]).to(self.device))
        self.episode_buffer[20] += copy.deepcopy(torch.tensor([[[(int(done))]]]).to(self.device))

    def save_next_observations(self, observations):
        node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask = observations
        self.episode_buffer[21] += copy.deepcopy(node_inputs)
        self.episode_buffer[22] += copy.deepcopy(edge_inputs)
        self.episode_buffer[23] += copy.deepcopy(enemy_inputs)
        self.episode_buffer[24] += copy.deepcopy(current_index)
        self.episode_buffer[25] += copy.deepcopy(node_padding_mask).bool()
        self.episode_buffer[26] += copy.deepcopy(edge_padding_mask).bool()
        self.episode_buffer[27] += copy.deepcopy(edge_mask).bool()
        self.episode_buffer[28] += copy.deepcopy(enemy_padding_mask).bool()

    def select_action_for_enemy(self, side):
        # 选取敌方动作
        reward = 0
        # ------------------------------------------------
        # 按攻击距离选择攻击和移动
        # # shoot_dist = 3
        # for enemy_idx, enemy_flag in enumerate(self.env.enemy_flags):
        #     if not enemy_flag:
        #         continue
        #     enemy_pos = self.env.enemy_indexes[enemy_idx]
        #     min_dist = float('inf')
        #     nearest_self_index = None

        #     # 最近的我方存活机器人
        #     for self_idx, self_flag in enumerate(self.env.self_flags):
        #         if self_flag:
        #             self_pos = self.env.self_indexes[self_idx]
        #             dist = self.env.network_adjacent_matrix[enemy_pos][self_pos]
        #             if dist < min_dist:
        #                 min_dist = dist
        #                 nearest_self_index = self_idx
        #     if nearest_self_index is not None:
        #     # if min_dist <= shoot_dist and nearest_self_index is not None:  # 攻击射击范围内的最近我方机器人
        #         reward += self.env.attack(attack_index=enemy_idx, defence_index=nearest_self_index, side=1)
        #         # print('enemy: ', enemy_idx, ' attack self: ', nearest_self_index)
        #         # print('reward:', reward)
        #     # elif nearest_self_index is not None:  # 向最近的我方机器人移动
        #     #     nearest_self_index = self.env.self_indexes[nearest_self_index]
        #     #     next_pos = self.env.next_node[enemy_pos][nearest_self_index]
        #     #     # print('enemy: ', enemy_idx, ' move from', enemy_pos, 'to: ', next_pos)
        #     #     reward += self.env.step(next_pos, enemy_idx, 1)
        # ------------------------------------------------
        # 按照命中率选择是否攻击
        if side == 1:
            for enemy_idx, enemy_flag in enumerate(self.env.enemy_flags):
                if not enemy_flag:
                    continue
                enemy_pos = self.env.enemy_indexes[enemy_idx]
                min_dist = float('inf')
                nearest_self_index = None

                # 最近的我方存活机器人
                for self_idx, self_flag in enumerate(self.env.self_flags):
                    if self_flag:
                        self_pos = self.env.self_indexes[self_idx]
                        dist = self.env.network_adjacent_matrix[enemy_pos][self_pos]
                        if dist < min_dist:
                            min_dist = dist
                            nearest_self_index = self_idx
                
                if nearest_self_index is not None:
                    nearest_self_pos = self.env.self_indexes[nearest_self_index]
                    attack_rate = self.env.damage[enemy_pos][nearest_self_pos]
                    attack_rate = attack_rate ** 2
                    random_num = random.random()
                    next_pos = self.env.next_node[enemy_pos][nearest_self_pos]
                    occupied = next_pos in self.env.self_indexes + self.env.enemy_indexes
                    if random_num < attack_rate or occupied:
                        reward += self.env.attack(attack_index=enemy_idx, defence_index=nearest_self_index, side=1)
                        # print('enemy: ', enemy_idx, ' attack self: ', nearest_self_index)
                    else:
                        reward += self.env.step(next_pos, enemy_idx, 1)
                        # print('enemy: ', enemy_idx, ' move from', enemy_pos, 'to: ', next_pos)
        else:
            for self_idx, self_flag in enumerate(self.env.self_flags):
                if not self_flag:
                    continue
                self_pos = self.env.self_indexes[self_idx]
                min_dist = float('inf')
                nearest_enemy_index = None

                # 最近的我方存活机器人
                for enemy_idx, enemy_flag in enumerate(self.env.enemy_flags):
                    if enemy_flag:
                        enemy_pos = self.env.enemy_indexes[enemy_idx]
                        dist = self.env.network_adjacent_matrix[self_pos][enemy_pos]
                        if dist < min_dist:
                            min_dist = dist
                            nearest_enemy_index = enemy_idx
                
                if nearest_enemy_index is not None:
                    nearest_enemy_pos = self.env.enemy_indexes[nearest_enemy_index]
                    attack_rate = self.env.damage[self_pos][nearest_enemy_pos]
                    attack_rate = attack_rate ** 2
                    random_num = random.random()
                    next_pos = self.env.next_node[self_pos][nearest_enemy_pos]
                    occupied = next_pos in self.env.self_indexes + self.env.enemy_indexes
                    if random_num < attack_rate or occupied:
                        reward += self.env.attack(attack_index=self_idx, defence_index=nearest_enemy_index, side=0)
                        # print('enemy: ', enemy_idx, ' attack self: ', nearest_self_index)
                    else:
                        reward += self.env.step(next_pos, self_idx, 0)
                        # print('enemy: ', enemy_idx, ' move from', enemy_pos, 'to: ', next_pos)
        return reward

    def handle_movement_conflicts(self, move_robots, move_targets, move_action_index_list, attack_robots, attack_targets, attack_action_index_list):
        action_index_list = np.zeros(len(move_action_index_list) + len(attack_action_index_list))
        # print('move_robots:', move_robots, 'move_targets:', move_targets, 'move_action_index_list:', move_action_index_list)
        # print('attack_robots:', attack_robots, 'attack_targets:', attack_targets, 'attack_action_index_list:', attack_action_index_list)
        # 处理移动冲突（包括敌我双方）
        target_to_robots = {}
        temp_move_robots = copy.deepcopy(move_robots)
        reward = 0
        for robot in temp_move_robots:
            target = move_targets[robot]
            # 检查目标位置是否已被其他机器人占用
            is_occupied = False
            for self_idx, self_pos in enumerate(self.env.self_indexes):
                if self_pos == target:
                    is_occupied = True
                    break
            for enemy_idx, enemy_pos in enumerate(self.env.enemy_indexes):
                if enemy_pos == target:
                    is_occupied = True
                    break
                    
            if not is_occupied:
                if target in target_to_robots:
                    target_to_robots[target].append(robot)
                else:
                    target_to_robots[target] = [robot]
            else:
                # 如果目标位置已被占用,从move_robots中移除该机器人
                move_robots.remove(robot)
                # print('remove robot:', robot)
        
        # print('target_to_robots:', target_to_robots, 'move_robots:', move_robots)
        # 对每个有冲突的目标位置随机选择一个机器人
        for target, robots in target_to_robots.items():
            if len(robots) > 1:
                # 随机选择一个机器人移动到该位置
                chosen_robot = random.choice(robots)
                # 从move_robots中移除未被选中的机器人
                for robot in robots:
                    if robot != chosen_robot:
                        move_robots.remove(robot)
        
        # 执行移动
        for i, robot in enumerate(move_robots):
            side, robot_index = robot
            next_index = move_targets[robot]
            action_index_list[robot_index] = move_action_index_list[i]
            # if side == 'self':
            #     print('self: ', robot_index, 'move from', self.env.self_indexes[robot_index], 'to: ', next_index)
            # else:
            #     print('enemy: ', robot_index, 'move from', self.env.enemy_indexes[robot_index], 'to: ', next_index)
            reward += self.env.step(next_index, robot_index, 0 if side == 'self' else 1)

        # 执行攻击
        for i, robot in enumerate(attack_robots):
            side, robot_index = robot
            target_index = attack_targets[robot]
            action_index_list[robot_index] = attack_action_index_list[i]
            # if side == 'self':
            #     print('self: ', robot_index, ' attack enemy: ', target_index)
            # else:
            #     print('enemy: ', robot_index, ' attack self: ', target_index)
            reward += self.env.attack(attack_index=robot_index, defence_index=target_index, side=0 if side == 'self' else 1)
        # print('action_index_list:', action_index_list)
        # print('reward:', reward)
        self.env.update_node_feature()
        
        return action_index_list, reward

    # def run_episode_random(self, curr_episode):
    #     done = False
        
    #     for i in range(128):
    #         print('--------------{}--------------'.format(i))
    #         reward = 0                        
    #         move_targets = {}  # 记录所有想移动的机器人及其目标位置
    #         move_robots = []   # 记录选择移动的机器人索引
            
    #         # self action
    #         # 收集我方机器人的移动意图
    #         for robot_index in range(N_SELF):
    #             random_num = random.random()
                
    #             if random_num < 0.5:  # move
    #                 observations, edge_for_select_node = self.get_observations(robot_index, side=0)
    #                 next_index, action_index = self.select_node_random(robot_index, observations, edge_for_select_node, side=0)
    #                 move_targets[('self', robot_index)] = next_index
    #                 move_robots.append(('self', robot_index))
    #             else:  # attack
    #                 candidate = [i for i, value in enumerate(self.env.enemy_flags) if value]
    #                 if candidate == []:
    #                     done = True
    #                     break
    #                 defence_index = random.choice(candidate)
    #                 reward = self.env.attack(attack_index=robot_index, defence_index=defence_index, side=0)
    #                 print('self: ', robot_index, ' attack enemy: ', defence_index)
            
    #         if done:
    #             self.env.stepi += 1
    #             if self.save_image:
    #                 if not os.path.exists(gifs_path):
    #                     os.makedirs(gifs_path)
    #                 self.env.plot_env(self.global_step, gifs_path, i)
    #             break
            
    #         # 收集敌方机器人的移动意图
    #         for robot_index in range(N_ENEMY):
    #             random_num = random.random()
                
    #             if random_num < 0.5:  # move
    #                 observations, edge_for_select_node = self.get_observations(robot_index, side=1)
    #                 next_index, action_index = self.select_node_random(robot_index, observations, edge_for_select_node, side=1)
    #                 move_targets[('enemy', robot_index)] = next_index
    #                 move_robots.append(('enemy', robot_index))
    #             else:  # attack
    #                 candidate = [i for i, value in enumerate(self.env.self_flags) if value]
    #                 if candidate == []:
    #                     done = True
    #                     break
    #                 defence_index = random.choice(candidate)
    #                 self.env.attack(attack_index=robot_index, defence_index=defence_index, side=1)
    #                 # print('enemy: ', robot_index, ' attack self: ', defence_index)
            
    #         # print('move_robots:', move_robots, 'move_targets:', move_targets)
            
    #         # 处理移动冲突
    #         self.handle_movement_conflicts(move_robots, move_targets)
            
    #         done, winner = self.env.check_done()
    #         self.env.stepi += 1

    #         # 检查所有机器人的位置是否重叠
    #         # 检查敌方内部不重合
    #         for robot_index in range(N_ENEMY):
    #             for robot_index2 in range(robot_index + 1, N_ENEMY):
    #                 if self.env.enemy_indexes[robot_index] == self.env.enemy_indexes[robot_index2]:
    #                     assert False, print('enemy: ', robot_index, 'and', robot_index2, 'have the same position')
    #         # 检查我方内部不重合
    #         for robot_index in range(N_SELF):
    #             for robot_index2 in range(robot_index + 1, N_SELF):
    #                 if self.env.self_indexes[robot_index] == self.env.self_indexes[robot_index2]:
    #                     assert False, print('self: ', robot_index, 'and', robot_index2, 'have the same position')
    #         # 检查敌我之间不重合
    #         for enemy_index in range(N_ENEMY):
    #             for self_index in range(N_SELF):
    #                 if self.env.enemy_indexes[enemy_index] == self.env.self_indexes[self_index]:
    #                     assert False, print('enemy: ', enemy_index, 'and self: ', self_index, 'have the same position')

    #         # save a frame
    #         if self.save_image:
    #             if not os.path.exists(gifs_path):
    #                 os.makedirs(gifs_path)
    #             self.env.plot_env(self.global_step, gifs_path, i)

    #         if done:
    #             break

    #     # save gif
    #     if self.save_image:
    #         path = gifs_path
    #         self.make_gif(path, curr_episode)

    def run_episode_random(self, curr_episode):
        done = False
        for i in range(128):
            # print('--------------{}--------------'.format(i))
            reward = 0   
            move_targets = {}  # 记录所有想移动的机器人及其目标位置
            move_robots = []   # 记录选择移动的机器人索引
            move_action_index_list = []
            attack_targets = {}  # 记录所有想攻击的机器人及其目标位置
            attack_robots = []   # 记录选择攻击的机器人索引
            attack_action_index_list = []
            # action_index_list = []

            # self action
            for robot_index in range(N_SELF):
                observations, edge_for_select_node = self.get_observations(robot_index, side=0)
                action_type, target_index, action_index = self.select_node_random(robot_index, observations, edge_for_select_node, side=0)
                
                if action_type == 'move':
                    move_targets[('self', robot_index)] = target_index
                    move_robots.append(('self', robot_index))
                    move_action_index_list.append(action_index)
                else:
                    attack_targets[('self', robot_index)] = target_index
                    attack_robots.append(('self', robot_index))
                    attack_action_index_list.append(action_index)

            # # enemy action
            # for robot_index in range(N_ENEMY):
            #     observations, edge_for_select_node = self.get_observations(robot_index, side=1)
            #     action_type, target_index, action_index = self.select_node_random(robot_index, observations, edge_for_select_node, side=1)
                
            #     if action_type == 'move':
            #         move_targets[('enemy', robot_index)] = target_index
            #         move_robots.append(('enemy', robot_index))
            #         move_action_index_list.append(action_index)
            #     else:
            #         attack_targets[('enemy', robot_index)] = target_index
            #         attack_robots.append(('enemy', robot_index))
            #         attack_action_index_list.append(action_index)

            # 处理移动冲突
            action_index_list, reward = self.handle_movement_conflicts(move_robots, move_targets, move_action_index_list, attack_robots, attack_targets, attack_action_index_list)
            print('action_index_list:', action_index_list)
            print('reward:', reward)
            done, winner = self.env.check_done()
            self.env.stepi += 1

            # 检查所有机器人的位置是否重叠
            # 检查敌方内部不重合
            for robot_index in range(N_ENEMY):
                for robot_index2 in range(robot_index + 1, N_ENEMY):
                    if self.env.enemy_indexes[robot_index] == self.env.enemy_indexes[robot_index2]:
                        assert False, print('enemy: ', robot_index, 'and', robot_index2, 'have the same position')
            # 检查我方内部不重合
            for robot_index in range(N_SELF):
                for robot_index2 in range(robot_index + 1, N_SELF):
                    if self.env.self_indexes[robot_index] == self.env.self_indexes[robot_index2]:
                        assert False, print('self: ', robot_index, 'and', robot_index2, 'have the same position')
            # 检查敌我之间不重合
            for enemy_index in range(N_ENEMY):
                for self_index in range(N_SELF):
                    if self.env.enemy_indexes[enemy_index] == self.env.self_indexes[self_index]:
                        assert False, print('enemy: ', enemy_index, 'and self: ', self_index, 'have the same position')

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i)

            if done:
                break

        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def run_episode_script_enemy(self, curr_episode):
        done = False
        self_win = False
        enemy_win = False
        for i in range(128):
            # print('--------------{}--------------'.format(i))
            reward = 0   
            action_index_list = []
            initial_all_observations, _ = self.get_all_observations(side=0)
            self.save_observations(initial_all_observations)
            # self action
            for robot_index in range(N_SELF):
                # choose move or attack
                observations, _ = self.get_observations(robot_index, side=0)
                action_type, target_index, action_index = self.select_action(observations)
                action_index_list.append(action_index)
                if action_type == 'move':
                    reward += self.env.step(target_index, robot_index, 0)
                else:
                    reward += self.env.attack(attack_index=robot_index, defence_index=target_index, side=0)
                
                self.env.update_node_feature()
            
            # enemy action
            reward +=self.select_action_for_enemy()

            self.env.update_node_feature()

            done, winner = self.env.check_done()
            self.env.stepi += 1

            self.save_action(action_index_list)
            self.save_reward_done(reward, done)

            final_all_observations, _ = self.get_all_observations(side=0)
            self.save_next_observations(final_all_observations)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i)

            if done:
                if winner == 0:
                    self_win = True
                    enemy_win = False
                elif winner == 1:
                    self_win = False
                    enemy_win = True
                else:
                    self_win = False
                    enemy_win = False
                break
            
        self.perf_metrics['success_rate'] = self_win
        self.perf_metrics['loss_rate'] = enemy_win
        self.perf_metrics['total_steps'] = i
        
        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def run_episode_script_enemy_sp(self, curr_episode):
        done = False
        self_win = False
        enemy_win = False
        for i in range(128):
            # print('--------------{}--------------'.format(i))
            reward = 0   
            self_action_index_list = []
            enemy_action_index_list = []

            initial_self_all_observations, _ = self.get_all_observations(side=0)
            
            # self action
            # reward += self.select_action_for_enemy(side=0)
            for robot_index in range(N_SELF):
                # choose move or attack
                observations, _ = self.get_observations(robot_index, side=0)
                action_type, target_index, action_index = self.select_action(observations, side=0)
                self_action_index_list.append(action_index)
                
                if action_type == 'move':
                    reward += self.env.step(target_index, robot_index, 0)
                else:
                    reward += self.env.attack(attack_index=robot_index, defence_index=target_index, side=0)
                
                self.env.update_node_feature()
            
            initial_enemy_all_observations, _ = self.get_all_observations(side=1)
            self.save_observations(initial_self_all_observations, initial_enemy_all_observations)

            # enemy action
            logp_list = []
            for robot_index in range(N_ENEMY):
                # if not self.env.enemy_flags[robot_index]:
                #     continue
                # choose move or attack
                observations, _ = self.get_observations(robot_index, side=1)
                action_type, target_index, action_index, logp = self.select_action(observations, side=1)
                
                enemy_action_index_list.append(action_index)
                logp_list.append(logp)
                if action_type == 'move':
                    reward += self.env.step(target_index, robot_index, 1)
                else:
                    reward += self.env.attack(attack_index=robot_index, defence_index=target_index, side=1)
                
                self.env.update_node_feature()

            # self.env.update_node_feature()

            done, winner = self.env.check_done()
            self.env.stepi += 1

            self.save_action(self_action_index_list, enemy_action_index_list)
            self.save_logp(logp_list)
            self.save_reward_done(reward, done)

            final_all_observations, _ = self.get_all_observations(side=0)
            self.save_next_observations(final_all_observations)

            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot_env(self.global_step, gifs_path, i)

            if done:
                if winner == 0:
                    self_win = 1
                    enemy_win = 0
                elif winner == 1:
                    self_win = 0
                    enemy_win = 1
                    i = 128
                else:
                    self_win = 0.5
                    enemy_win = 0.5
                break
        
        if not done:
            self_win = 0.5
            enemy_win = 0.5

        self.perf_metrics['success_rate'] = self_win
        self.perf_metrics['loss_rate'] = enemy_win
        self.perf_metrics['total_steps'] = i
        
        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def generate_batch_data(self):
        # 随机初始状态
        # 位置
        enemy_index = np.random.choice(self.env.node_num, N_ENEMY)
        self_indexes = np.random.choice(self.env.node_num, N_SELF)
        self.env.target_index = list(enemy_index)
        self.env.robot_indexes = list(self_indexes)
        print('target_index: ', self.env.target_index, 'robot_indexes: ', self.env.robot_indexes)
        # 存活状态
        self.env.self_flags = [True] * N_SELF
        self.env.enemy_flags = np.random.choice([True, False], N_ENEMY).tolist()
        print('self_flags: ', self.env.self_flags, 'enemy_flags: ', self.env.enemy_flags)
        # 血量
        self.env.self_bloods = [MAX_BLOOD] * N_SELF
        self.env.enemy_bloods = [np.random.uniform(0, MAX_BLOOD) if flag else 0.0 
                       for flag in self.env.enemy_flags]
        print('self_bloods: ', self.env.self_bloods, 'enemy_bloods: ', self.env.enemy_bloods)
        # 生成对应的节点特征
        self.env.update_node_feature()

        reward = 0   
        move_targets = {}  # 记录所有想移动的机器人及其目标位置
        move_robots = []   # 记录选择移动的机器人索引
        move_action_index_list = []
        attack_targets = {}  # 记录所有想攻击的机器人及其目标位置
        attack_robots = []   # 记录选择攻击的机器人索引
        attack_action_index_list = []
        action_index_list = []
        initial_all_observations, _ = self.get_all_observations(side=0)
        self.save_observations(initial_all_observations)

        for robot_index in range(N_SELF):
            observations, _ = self.get_observations(robot_index, side=0)
            action_type, target_index, action_index = self.select_action(observations)
            if action_type == 'move':
                move_targets[('self', robot_index)] = target_index
                move_robots.append(('self', robot_index))
                move_action_index_list.append(action_index)
            else:
                attack_targets[('self', robot_index)] = target_index
                attack_robots.append(('self', robot_index))
                attack_action_index_list.append(action_index)
                
        # 处理移动冲突
        action_index_list, reward = self.handle_movement_conflicts(move_robots, move_targets, move_action_index_list, attack_robots, attack_targets, attack_action_index_list)

        done, winner = self.env.check_done()
        
        self.save_action(action_index_list)
        self.save_reward_done(reward, done)

        final_all_observations, _ = self.get_all_observations(side=0)
        self.save_next_observations(final_all_observations)

    def work(self, currEpisode):
        # self.run_episode_random(currEpisode)
        self.run_episode_script_enemy_sp(currEpisode)

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_step_{:.4g}.gif'.format(path, n, self.env.stepi), mode='I', fps=2) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files:
            os.remove(filename)



