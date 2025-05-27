import imageio
import csv
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import Env
from model import PolicyNet
from test_parameter import *
import json
import networkx as nx
import random

class TestWorker:
    def __init__(self, meta_agent_id, policy_net_self, policy_net_enemy, global_step, device='cuda', greedy=False, save_image=False, random_seed=None):
        self.device = device
        self.greedy = greedy
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(k_size=self.k_size, plot=save_image, test=True, num_self=N_SELF, random_seed=random_seed)
        self.local_policy_net_self = policy_net_self
        self.local_policy_net_enemy = policy_net_enemy
        # self.travel_dist = 0
        self.perf_metrics = dict()

    def run_episode_script_enemy(self, curr_episode):
        done = False
        self_win = False
        enemy_win = False

        episode_actions = {
            'self_actions': [],  # 我方动作记录
            'enemy_actions': [],  # 敌方动作记录
            'self_indexes': [],  # 我方位置记录
            'enemy_indexes': [],  # 敌方位置记录
            'self_flags': [],  # 我方存活状态
            'enemy_flags': [],  # 敌方存活状态
            'self_bloods': [],  # 我方血量
            'enemy_bloods': []  # 敌方血量
        }

        for i in range(128):
            # print('--------------{}--------------'.format(i))   
            episode_actions['self_indexes'].append(self.env.self_indexes.copy())
            episode_actions['enemy_indexes'].append(self.env.enemy_indexes.copy())
            episode_actions['self_flags'].append(self.env.self_flags.copy())
            episode_actions['enemy_flags'].append(self.env.enemy_flags.copy())
            episode_actions['self_bloods'].append(self.env.self_bloods.copy())
            episode_actions['enemy_bloods'].append(self.env.enemy_bloods.copy())
            
            # 记录这个时间步的动作
            step_self_actions = []
            step_enemy_actions = []
         
            # self action
            for robot_index in range(N_SELF):
                # choose move or attack
                observations = self.get_observations(robot_index, side=0)
                action_type, target_index, action_index = self.select_action(observations, side=0)
                # print('robot_index: ', robot_index, 'action_type: ', action_type, 'target_index: ', target_index, 'action_index: ', action_index)
                # 记录动作
                step_self_actions.append({
                    'robot_index': robot_index,
                    'action_type': action_type,
                    'target_index': target_index
                })
                
                # if action_type == 'move':
                #     self.env.step(target_index, robot_index, 0)
                # else:
                #     self.env.attack(attack_index=robot_index, defence_index=target_index, side=0)
                # self.env.update_node_feature()
            # enemy action
            # self.select_action_for_enemy()
            for robot_index in range(N_ENEMY):
                # choose move or attack
                observations = self.get_observations(robot_index, side=1)
                action_type, target_index, action_index = self.select_action(observations, side=1)
                # print('robot_index: ', robot_index, 'action_type: ', action_type, 'target_index: ', target_index, 'action_index: ', action_index)
                
                step_enemy_actions.append({
                    'robot_index': robot_index,
                    'action_type': action_type,
                    'target_index': target_index
                })

                # if action_type == 'move':
                #     self.env.step(target_index, robot_index, 1)
                # else:
                #     self.env.attack(attack_index=robot_index, defence_index=target_index, side=1)
                # self.env.update_node_feature()
            
            episode_actions['self_actions'].append(step_self_actions)
            episode_actions['enemy_actions'].append(step_enemy_actions)

            self.env.update_node_feature()
            done, winner = self.env.check_done()
            self.env.stepi += 1

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

                # if winner == 0:
                #     self_win = 1
                #     enemy_win = 0
                # elif winner == 1:
                #     self_win = 0
                #     enemy_win = 1
                #     # i = 128
                # else:
                #     self_win = 0
                #     enemy_win = 0
                # break
            
        if not done:
            self_win = 0.5
            enemy_win = 0.5
        
        episode_actions['self_indexes'].append(self.env.self_indexes.copy())
        episode_actions['enemy_indexes'].append(self.env.enemy_indexes.copy())
        episode_actions['self_flags'].append(self.env.self_flags.copy())
        episode_actions['enemy_flags'].append(self.env.enemy_flags.copy())
        episode_actions['self_bloods'].append(self.env.self_bloods.copy())
        episode_actions['enemy_bloods'].append(self.env.enemy_bloods.copy())

        # 保存动作记录
        if SAVE_TRAJECTORY:
            if not os.path.exists(trajectory_path):
                os.makedirs(trajectory_path)
            with open(f'{trajectory_path}/episode_{curr_episode}_actions.json', 'w') as f:
                json.dump(episode_actions, f)

        self.perf_metrics['success_rate'] = self_win
        self.perf_metrics['loss_rate'] = enemy_win
        self.perf_metrics['steps'] = i
        
        # save gif
        if self.save_image:
            path = gifs_path
            self.make_gif(path, curr_episode)

    def get_observations(self, robot_index, side):
        
        # get observations
        graph = self.env.graph
        if side == 0:
            node_feature = self.env.node_feature_self
        else:
            node_feature = self.env.node_feature_enemy
        # print('node feature: ', node_feature)
        # normalize observations
        # node_feature = node_feature / np.max(self.env.network_adjacent_matrix)
        # print('network adj matrix: ', self.env.network_adjacent_matrix)
        # transfer to node inputs tensor
        node_feature_inputs = node_feature.reshape((self.env.node_num, -1))
        # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
        node_inputs = node_feature_inputs
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

        # get the node index of the current robot position
        enemy_input = []
        if side == 0:
            current_node_index = self.env.self_indexes[robot_index]
            for i in range(N_ENEMY):
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

        # calculate a mask for padded node
        node_padding_mask = None

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        edge_inputs = []
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors = [n for n in neighbors if n not in self.env.self_indexes + self.env.enemy_indexes]
            if node in neighbors:
                neighbors.remove(node)
            edge_list = [node] + neighbors  # 再将当前节点放在最前面
            edge_inputs.append(edge_list)

        adjacent_matrix = self.env.adjacent_matrix
        # print('adj matrix: ', adjacent_matrix)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)

        edge = edge_inputs[current_index]
        while len(edge) < self.k_size:
            edge.append(-1)

        edge_input = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)

        # calculate a mask for the padded edges (denoted by -1)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        edge_padding_mask = torch.where(edge_input == -1, one, edge_padding_mask)

        edge_inputs = torch.where(edge_input == -1, 0, edge_input)
        observations = node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask
        
        return observations
    
    def get_all_observations(self):
        # get observations
        node_coords = self.env.node_coords
        graph = self.env.graph
        node_feature = self.env.node_feature
        
        # normalize observations
        node_feature = node_feature / self.env.graph_generator.max_dist

        # transfer to node inputs tensor
        n_nodes = node_coords.shape[0]
        node_feature_inputs = node_feature.reshape((n_nodes, N_SELF + 1))
        # permutation = [0] + [(i + robot_index) % N_ROBOTS + 1 for i in range(N_ROBOTS)]
        node_inputs = node_feature_inputs
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_size, 4)

        # get the node index of the current robot position
        all_current_indexes = []
        for i in range(N_SELF):
            all_current_node_index = self.env.find_index_from_coords(self.robot_positions[i])
            all_current_indexes.append(all_current_node_index)
        all_current_index = torch.tensor(all_current_indexes).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1, N_ROBOTS)
        
        # calculate a mask for padded node
        node_padding_mask = None

        # prepare the adjacent list as padded edge inputs and the adjacent matrix as the edge mask
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        adjacent_matrix = self.env.adjacent_matrix

        # edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        edge_mask = torch.from_numpy(adjacent_matrix).float().unsqueeze(0).to(self.device)
        all_edges_inputs = []
        all_edge_padding_masks = []
        for i in range(N_SELF):
            edge = edge_inputs[all_current_indexes[i]]
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
        all_observations = node_inputs, all_edge_inputs, all_current_index, node_padding_mask, all_edge_padding_mask, edge_mask

        return all_observations
    
    # def select_action(self, observations):
    #     node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask = observations
    #     with torch.no_grad():
    #         logp = self.local_policy_net(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
    #                                             edge_padding_mask, edge_mask, enemy_padding_mask)
    #     if self.greedy:
    #         action_index = torch.argmax(logp, dim=1).long()
    #     else:
    #         action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
    #     action_index = action_index.item()

    #     # 判断选择的是移动还是攻击
    #     if action_index < K_SIZE:  # 选择移动
    #         # 获取目标节点索引
    #         target_index = edge_inputs[0, 0, action_index].item()
    #         # print('robot_index: ', current_index, 'move: ', target_index)
    #         return 'move', target_index, action_index
    #     else:  # 选择攻击
    #         # 将action_index转换为敌人索引
    #         attack_index = action_index - K_SIZE
    #         # 获取可攻击的敌人列表
    #         candidate = []
    #         for i in range(N_ENEMY):
    #             if self.env.enemy_flags[i] and self.env.shoot_matrix[current_index][self.env.enemy_indexes[i]] == 1:
    #                 candidate.append(i)
    #         target_index = candidate[attack_index]
    #         # print('robot_index: ', current_index, 'attack: ', target_index)
    #         return 'attack', target_index, action_index
        
    def select_action(self, observations, side):
        node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask, enemy_padding_mask = observations
        # print('edge_inputs: ', edge_inputs.shape, 'current_index: ', current_index.shape, 'enemy_inputs: ', enemy_inputs.shape, 'edge_padding_mask: ', edge_padding_mask.shape, 'enemy_padding_mask: ', enemy_padding_mask.shape)
        if side == 0:
            with torch.no_grad():
                logp = self.local_policy_net_self(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
                                                    edge_padding_mask, edge_mask, enemy_padding_mask)
            #     log_ref = self.local_policy_net_enemy(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
            #                                         edge_padding_mask, edge_mask, enemy_padding_mask)
                
            # # 求散度
            # kl_div = torch.nn.functional.kl_div(logp, log_ref, reduction='batchmean')
            # print('kl_div: ', kl_div)
            if self.greedy:
                action_index = torch.argmax(logp, dim=1).long()
            else:
                action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
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
                logp = self.local_policy_net_enemy(node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask,
                                                    edge_padding_mask, edge_mask, enemy_padding_mask)
            
            if self.greedy:
                action_index = torch.argmax(logp, dim=1).long()
            else:
                action_index = torch.multinomial(logp.exp(), 1).long().squeeze(1)
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
                for i in range(N_SELF):
                    if self.env.self_flags[i] and self.env.shoot_matrix[current_index][self.env.self_indexes[i]] == 1:
                        candidate.append(i)
                target_index = candidate[attack_index]
                return 'attack', target_index, action_index
    
    def select_action_for_enemy(self):
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
        return reward
    
    def handle_movement_conflicts(self, move_robots, move_targets, move_action_index_list, attack_robots, attack_targets, attack_action_index_list):
        action_index_list = np.zeros(len(move_action_index_list) + len(attack_action_index_list))
        # print('move_robots:', move_robots, 'move_targets:', move_targets, 'move_action_index_list:', move_action_index_list)
        # print('attack_robots:', attack_robots, 'attack_targets:', attack_targets, 'attack_action_index_list:', attack_action_index_list)
        # 处理移动冲突（包括敌我双方）
        target_to_robots = {}
        temp_move_robots = copy.deepcopy(move_robots)
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
            self.env.step(next_index, robot_index, 0 if side == 'self' else 1)

        # 执行攻击
        for i, robot in enumerate(attack_robots):
            side, robot_index = robot
            target_index = attack_targets[robot]
            action_index_list[robot_index] = attack_action_index_list[i]
            # if side == 'self':
            #     print('self: ', robot_index, ' attack enemy: ', target_index)
            # else:
            #     print('enemy: ', robot_index, ' attack self: ', target_index)
            self.env.attack(attack_index=robot_index, defence_index=target_index, side=0 if side == 'self' else 1)
        # print('action_index_list:', action_index_list)
        # print('move_robots:', move_robots, 'move_targets:', move_targets)
        # print('attack_robots:', attack_robots, 'attack_targets:', attack_targets)
        self.env.update_node_feature()
        
        return action_index_list

    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_step_{:.4g}.gif'.format(path, n, self.perf_metrics['steps']), mode='I', fps=2) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files:
            os.remove(filename)


    def work(self, curr_episode):
        self.run_episode_script_enemy(curr_episode)