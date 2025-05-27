import matplotlib.pyplot as plt
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np

from parameter import *
if not train_mode:
    from test_parameter import *

np.set_printoptions(threshold=np.inf)

class Env():
    def __init__(self, num_self=N_SELF, num_enemy=N_ENEMY, random_seed=None, k_size=20, plot=False, test=False):
        # import environment ground truth from dungeon files
        self.test = test
        self.num_self = num_self
        self.num_enemy = num_enemy
        self.random_seed = random_seed
        
        self.adj_dir = adj_path

        adjacent_matrix = np.loadtxt(self.adj_dir + '/connectivity.txt', dtype=int)
        self.shoot_matrix = np.loadtxt(self.adj_dir + '/can_shoot.txt', dtype=int)
        location = json.load(open(self.adj_dir + "/location_of_points.json"))

        self.node_coords = np.array(list(location.values()))
        self.node_num = self.node_coords.shape[0]

        self.self_bloods = []
        self.self_flags = []
        for k in range(num_self):
            self.self_bloods.append(MAX_BLOOD)
            self.self_flags.append(True)
        
        self.enemy_bloods = []
        self.enemy_flags = []
        for k in range(num_enemy):
            self.enemy_bloods.append(MAX_BLOOD)
            self.enemy_flags.append(True)

        self.graph, self.adjacent_matrix, self.network_adjacent_matrix, self.max_dist, self.node_feature_self, self.node_feature_enemy, \
            self.real_dist, self.damage, self.self_indexes, self.enemy_indexes, self.start_self_positions, \
                self.start_enemy_positions, self.next_node = self.import_adj_matrix(adjacent_matrix)
        
        self.self_positions = []
        for k in range(num_self):
            self.self_positions.append(self.start_self_positions[k])
        
        self.enemy_positions = []
        for k in range(num_enemy):
            self.enemy_positions.append(self.start_enemy_positions[k])

        self.stepi = 0

        # plot related
        self.plot = plot
        self.frame_files = []
        self.self_points = {}
        self.enemy_points = {}
        if self.plot:
            # initialize the route
            for i in range(self.num_self):
                self.self_points['x'+str(i)] = [self.start_self_positions[i][0]]
                self.self_points['y'+str(i)] = [self.start_self_positions[i][1]]
            for i in range(self.num_enemy):
                self.enemy_points['x'+str(i)] = [self.start_enemy_positions[i][0]]
                self.enemy_points['y'+str(i)] = [self.start_enemy_positions[i][1]]

    def import_adj_matrix(self, adj_matrix):
        graph = nx.from_numpy_array(adj_matrix)
        adj_matrix = 1 - adj_matrix
        assert np.array_equal(adj_matrix, adj_matrix.T)

        network_adjacent_matrix = nx.floyd_warshall_numpy(graph)
        max_dist = np.max(network_adjacent_matrix)

        next_node = [[j for j in range(self.node_num)] for i in range(self.node_num)]
        for i in range(self.node_num):
            for j in range(self.node_num):
                for k in range(self.node_num):
                    if network_adjacent_matrix[i][k] == 1 and network_adjacent_matrix[i][j] == network_adjacent_matrix[k][j] + 1:
                        next_node[i][j] = k
        next_node = np.array(next_node)

        # enemy_indexes = random.sample(range(self.node_num), self.num_enemy)
        enemy_indexes = self.select_close_positions(range(self.node_num), 
                                            network_adjacent_matrix, 
                                            self.num_enemy, 
                                            max_distance=3)
        candidate_self_indexes = [i for i in range(self.node_num) if i not in enemy_indexes]
        # self_indexes = random.sample(candidate_self_indexes, self.num_self)
        self_indexes = self.select_close_positions(candidate_self_indexes, 
                                            network_adjacent_matrix, 
                                            self.num_self, 
                                            max_distance=3)

        enemy_positions = []
        for i in enemy_indexes:
            enemy_position = self.node_coords[i]
            enemy_positions.append(enemy_position)
        
        self_positions = []
        for i in self_indexes:
            self_position = self.node_coords[i]
            self_positions.append(self_position)

        diff = self.node_coords[:, np.newaxis, :] - self.node_coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))/10

        damage = np.zeros((self.node_num, self.node_num))
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i != j and self.shoot_matrix[i][j] == 1:
                    dist = distances[i][j]
                    damage[i][j] = -0.1519 * dist + 1.1808  # y = -0.1519x + 1.1808
                    damage[i][j] = np.clip(damage[i][j], 0, 1)
                if i == j:
                    damage[i][j] = 1
        
        node_feature_self = np.zeros((self.node_num, len(enemy_indexes)*2 + len(self_indexes)*2 + 2 + 2))
        for index in range(self.node_num):
            for i, robot in enumerate(enemy_indexes):
                node_feature_self[index][i] = network_adjacent_matrix[robot][index]/max_dist
            for i, robot in enumerate(self_indexes):
                node_feature_self[index][len(enemy_indexes) + i] = network_adjacent_matrix[robot][index]/max_dist
            for i, robot in enumerate(enemy_indexes):
                node_feature_self[index][len(enemy_indexes) + len(self_indexes) + i] = damage[index][robot]
            for i, robot in enumerate(self_indexes):
                node_feature_self[index][len(enemy_indexes)*2 + len(self_indexes) + i] = damage[index][robot]
        
        for i, robot in enumerate(enemy_indexes):
            node_feature_self[robot][len(enemy_indexes)*2 + len(self_indexes)*2] = 1
        for i, robot in enumerate(self_indexes):
            node_feature_self[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 1] = 1
      
        for i, robot in enumerate(enemy_indexes):
            node_feature_self[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 2] = self.enemy_bloods[i]/MAX_BLOOD
        for i, robot in enumerate(self_indexes):
            node_feature_self[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 3] = self.self_bloods[i]/MAX_BLOOD
          
        node_feature_enemy = np.zeros((self.node_num, len(enemy_indexes)*2 + len(self_indexes)*2 + 2 + 2))
        for index in range(self.node_num):
            for i, robot in enumerate(self_indexes):
                node_feature_enemy[index][i] = network_adjacent_matrix[robot][index]/max_dist
            for i, robot in enumerate(enemy_indexes):
                node_feature_enemy[index][len(enemy_indexes) + i] = network_adjacent_matrix[robot][index]/max_dist
            for i, robot in enumerate(self_indexes):
                node_feature_enemy[index][len(enemy_indexes) + len(self_indexes) + i] = damage[index][robot]
            for i, robot in enumerate(enemy_indexes):
                node_feature_enemy[index][len(enemy_indexes)*2 + len(self_indexes) + i] = damage[index][robot]

        for i, robot in enumerate(self_indexes):
            node_feature_enemy[robot][len(enemy_indexes)*2 + len(self_indexes)*2] = 1
        for i, robot in enumerate(enemy_indexes):
            node_feature_enemy[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 1] = 1
      
        for i, robot in enumerate(self_indexes):
            node_feature_enemy[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 2] = self.self_bloods[i]/MAX_BLOOD
        for i, robot in enumerate(enemy_indexes):
            node_feature_enemy[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 3] = self.enemy_bloods[i]/MAX_BLOOD

        return graph, adj_matrix, network_adjacent_matrix, max_dist, node_feature_self, node_feature_enemy, distances, damage, self_indexes, enemy_indexes, self_positions, enemy_positions, next_node

    def reset(self):
        self.self_bloods = []
        self.self_flags = []
        for k in range(self.num_self):
            self.self_bloods.append(MAX_BLOOD)
            self.self_flags.append(True)
        
        self.enemy_bloods = []
        self.enemy_flags = []
        for k in range(self.num_enemy):
            self.enemy_bloods.append(MAX_BLOOD)
            self.enemy_flags.append(True)
        
        enemy_indexes = random.sample(range(self.node_num), self.num_enemy)
        candidate_self_indexes = [i for i in range(self.node_num) if i not in enemy_indexes]
        # self_indexes = random.sample(candidate_self_indexes, self.num_self)

        self_indexes = self.select_close_positions(candidate_self_indexes, 
                                            self.network_adjacent_matrix, 
                                            self.num_self, 
                                            max_distance=3)

        enemy_positions = []
        for i in enemy_indexes:
            enemy_position = self.node_coords[i]
            enemy_positions.append(enemy_position)
        
        self_positions = []
        for i in self_indexes:
            self_position = self.node_coords[i]
            self_positions.append(self_position)

        self.last_closest_dist = [
            [self.network_adjacent_matrix[i, j] for j in enemy_indexes]
            for i in self_indexes
        ]
        
        node_feature = np.zeros((self.node_num, len(enemy_indexes)*2 + len(self_indexes)*2 + 2 + 2))
        for index in range(self.node_num):
            for i, robot in enumerate(enemy_indexes):
                node_feature[index][i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(self_indexes):
                node_feature[index][len(enemy_indexes) + i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(enemy_indexes):
                node_feature[index][len(enemy_indexes) + len(self_indexes) + i] = self.damage[index][robot]
            for i, robot in enumerate(self_indexes):
                node_feature[index][len(enemy_indexes)*2 + len(self_indexes) + i] = self.damage[index][robot]
        
        for i, robot in enumerate(enemy_indexes):
            node_feature[robot][len(enemy_indexes)*2 + len(self_indexes)*2] = 1
        for i, robot in enumerate(self_indexes):
            node_feature[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 1] = 1
      
        for i, robot in enumerate(enemy_indexes):
            node_feature[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 2] = self.enemy_bloods[i]/MAX_BLOOD
        for i, robot in enumerate(self_indexes):
            node_feature[robot][len(enemy_indexes)*2 + len(self_indexes)*2 + 3] = self.self_bloods[i]/MAX_BLOOD

        self.node_feature = node_feature
        self.self_indexes = self_indexes
        self.enemy_indexes = enemy_indexes
        self.start_self_positions = self_positions
        self.start_enemy_positions = enemy_positions
        
        self.self_positions = []
        for k in range(self.num_self):
            self.self_positions.append(self.start_self_positions[k])
        
        self.enemy_positions = []
        for k in range(self.num_enemy):
            self.enemy_positions.append(self.start_enemy_positions[k])

        self.stepi = 0

        # plot related

        self.frame_files = []
        self.self_points = {}
        self.enemy_points = {}
        if self.plot:
            # initialize the route
            for i in range(self.num_self):
                self.self_points['x'+str(i)] = [self.start_self_positions[i][0]]
                self.self_points['y'+str(i)] = [self.start_self_positions[i][1]]
            for i in range(self.num_enemy):
                self.enemy_points['x'+str(i)] = [self.start_enemy_positions[i][0]]
                self.enemy_points['y'+str(i)] = [self.start_enemy_positions[i][1]]

    def step(self, next_node_index, robot_index, side):
        if side != 0 and side != 1:
            raise ValueError("Ensure side=0 or side=1")
        
        if side == 0:
            if self.self_flags[robot_index]:
                next_position = self.node_coords[next_node_index]
                self.self_indexes[robot_index] = int(next_node_index)
                self.self_positions[robot_index] = next_position

            if self.plot:
                self.self_points['x'+str(robot_index)].append(self.self_positions[robot_index][0])
                self.self_points['y'+str(robot_index)].append(self.self_positions[robot_index][1])

            # self.update_node_feature()
            
        else:
            if self.enemy_flags[robot_index]:
                next_position = self.node_coords[next_node_index]
                self.enemy_indexes[robot_index] = int(next_node_index)
                self.enemy_positions[robot_index] = next_position

            if self.plot:
                self.enemy_points['x'+str(robot_index)].append(self.enemy_positions[robot_index][0])
                self.enemy_points['y'+str(robot_index)].append(self.enemy_positions[robot_index][1])
            
            # self.update_node_feature()

        reward = 0
        
        return reward

    def update_node_feature(self):
        node_feature_self = np.zeros((self.node_num, self.num_enemy*2 + self.num_self*2 + 2 + 2))
        node_feature_enemy = np.zeros((self.node_num, self.num_enemy*2 + self.num_self*2 + 2 + 2))
        for index in range(self.node_num):
            for i, robot in enumerate(self.enemy_indexes):
                node_feature_self[index][i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(self.self_indexes):
                node_feature_self[index][self.num_enemy + i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(self.enemy_indexes):
                if self.enemy_flags[i]:
                    node_feature_self[index][self.num_enemy + self.num_self + i] = self.damage[index][robot]
                else:
                    node_feature_self[index][self.num_enemy + self.num_self + i] = 0
            for i, robot in enumerate(self.self_indexes):
                if self.self_flags[i]:
                    node_feature_self[index][self.num_enemy*2 + self.num_self + i] = self.damage[index][robot]
                else:
                    node_feature_self[index][self.num_enemy*2 + self.num_self + i] = 0
        
        for i, robot in enumerate(self.enemy_indexes):
            node_feature_self[robot][self.num_enemy*2 + self.num_self*2] = 1
        for i, robot in enumerate(self.self_indexes):
            node_feature_self[robot][self.num_enemy*2 + self.num_self*2 + 1] = 1

        for i, robot in enumerate(self.enemy_indexes):
            if self.enemy_flags[i]:
                node_feature_self[robot][self.num_enemy*2 + self.num_self*2 + 2] = self.enemy_bloods[i]/MAX_BLOOD
            else:
                node_feature_self[robot][self.num_enemy*2 + self.num_self*2 + 2] = -1
        for i, robot in enumerate(self.self_indexes):
            if self.self_flags[i]:
                node_feature_self[robot][self.num_enemy*2 + self.num_self*2 + 3] = self.self_bloods[i]/MAX_BLOOD
            else:
                node_feature_self[robot][self.num_enemy*2 + self.num_self*2 + 3] = -1
        
        for index in range(self.node_num):
            for i, robot in enumerate(self.self_indexes):
                node_feature_enemy[index][i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(self.enemy_indexes):
                node_feature_enemy[index][self.num_enemy + i] = self.network_adjacent_matrix[robot][index]/self.max_dist
            for i, robot in enumerate(self.self_indexes):
                if self.self_flags[i]:
                    node_feature_enemy[index][self.num_enemy + self.num_self + i] = self.damage[index][robot]
                else:
                    node_feature_enemy[index][self.num_enemy + self.num_self + i] = 0
            for i, robot in enumerate(self.enemy_indexes):
                if self.enemy_flags[i]:
                    node_feature_enemy[index][self.num_enemy*2 + self.num_self + i] = self.damage[index][robot]
                else:
                    node_feature_enemy[index][self.num_enemy*2 + self.num_self + i] = 0
        
        for i, robot in enumerate(self.self_indexes):
            node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2] = 1
        for i, robot in enumerate(self.enemy_indexes):
            node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2 + 1] = 1

        for i, robot in enumerate(self.self_indexes):
            if self.self_flags[i]:
                node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2 + 2] = self.self_bloods[i]/MAX_BLOOD
            else:
                node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2 + 2] = -1
        for i, robot in enumerate(self.enemy_indexes):
            if self.enemy_flags[i]:
                node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2 + 3] = self.enemy_bloods[i]/MAX_BLOOD
            else:
                node_feature_enemy[robot][self.num_enemy*2 + self.num_self*2 + 3] = -1
                
        self.node_feature_self = node_feature_self
        self.node_feature_enemy = node_feature_enemy

    def attack(self, attack_index, defence_index, side):
        reward = 0
        
        if side == 0:
            if self.enemy_flags[defence_index] == False:
                return 0
            
            if self.self_flags[attack_index]:
                damage = self.damage[self.self_indexes[attack_index]][self.enemy_indexes[defence_index]]
                # print('damage: ', damage)

                if self.shoot_matrix[self.self_indexes[attack_index]][self.enemy_indexes[defence_index]] == 0:
                    assert damage == 0, "When shoot_matrix==0, damage must be 0."

                pre_die_num = self.enemy_flags.count(False)

                self.enemy_bloods[defence_index] -= damage
                self.enemy_bloods[defence_index] = max(self.enemy_bloods[defence_index], 0)
                if self.enemy_bloods[defence_index] <= 0:
                    self.enemy_flags[defence_index] = False
                    # self.update_next_node()

                curr_die_num = self.enemy_flags.count(False)
                reward = self.calculate_reward(damage=damage, pre_die_num=pre_die_num, curr_die_num=curr_die_num, side=0)
        
        elif side == 1:
            if self.self_flags[defence_index] == False:
                return 0
            
            if self.enemy_flags[attack_index]:
                damage = self.damage[self.enemy_indexes[attack_index]][ self.self_indexes[defence_index]]

                if self.shoot_matrix[self.enemy_indexes[attack_index]][self.self_indexes[defence_index]] == 0:
                    assert damage == 0, "When shoot_matrix==0, damage must be 0."

                pre_die_num = self.self_flags.count(False)

                self.self_bloods[defence_index] -= damage
                self.self_bloods[defence_index] = max(self.self_bloods[defence_index], 0)
                if self.self_bloods[defence_index] <= 0:
                    self.self_flags[defence_index] = False
                    # self.update_next_node()
            
                curr_die_num = self.self_flags.count(False)
                reward = self.calculate_reward(damage=damage, pre_die_num=pre_die_num, curr_die_num=curr_die_num, side=1)

        else:
            raise ValueError("Ensure side=0 or side=1")

        return reward
    
    def update_next_node(self):
        adj_matrix = 1 - self.adjacent_matrix
        self_dead_indexes = [i for i, flag in enumerate(self.self_flags) if not flag]
        self_dead_node = [self.self_indexes[i] for i in self_dead_indexes]
        enemy_dead_indexes = [i for i, flag in enumerate(self.enemy_flags) if not flag]
        enemy_dead_node = [self.enemy_indexes[i] for i in enemy_dead_indexes]

        dead_node = self_dead_node + enemy_dead_node

        for i in dead_node:
            adj_matrix[i] = 0
            adj_matrix[:, i] = 0
       
        # ----------------------------------------------------------------------------
        dist = [[9999 for i in range(self.node_num)] for j in range(self.node_num)]
        next_node = [[j for j in range(self.node_num)] for i in range(self.node_num)]
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    dist[i][j] = 0
                elif adj_matrix[i][j] == 1:
                    dist[i][j] = 1

        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
        dist = np.array(dist)
        next_node = np.array(next_node)
        # ----------------------------------------------------------------------------
        self.network_adjacent_matrix = dist
        self.net_node_matrix = next_node

    def check_done(self):
        done = False
        winner = None  # 0: self win, 1: enemy win        
        if self.enemy_flags.count(False) == N_ENEMY:
            done = True
            winner = 0
        elif self.self_flags.count(False) == N_SELF:
            done = True
            winner = 1
            
        return done, winner

    def calculate_reward(self, damage=0, pre_die_num=0, curr_die_num=0, side=0):
        if side == 0:
            reward = (curr_die_num - pre_die_num) * 3
            if curr_die_num == N_ENEMY:
                reward += 20

        elif side == 1:
            reward = 0

            # if curr_die_num == N_SELF:
            #     reward -= 20

        return reward

    def plot_env(self, n, path, step):
        colors_self = ['r', 'g', 'y']
        colors_enemy = ['purple', 'c', 'coral']

        plt.figure(figsize=(10,6))
        plt.scatter(self.node_coords[:,0], self.node_coords[:,1],c='none',marker='o',s=50,edgecolors='#696969')
        for i in range(len(self.adjacent_matrix)):
            for j in range(i + 1, len(self.adjacent_matrix)):
                if self.adjacent_matrix[i, j] == 0 or self.adjacent_matrix[j, i] == 0:
                    plt.plot([self.node_coords[i, 0], self.node_coords[j, 0]],
                            [
                            self.node_coords[i, 1], self.node_coords[j, 1]], color='#C0C0C0')

        for i, (x, y) in enumerate(self.node_coords):
            plt.text(x+0.5, y+0.8, f'{i}', fontsize=11)

        for i in range(self.num_self):
            if self.self_flags[i]:
                plt.scatter(self.self_points['x'+str(i)][-1], self.self_points['y'+str(i)][-1], c=colors_self[i], s=250-i*70, zorder=11)
            else:
                plt.scatter(self.self_points['x'+str(i)][-1], self.self_points['y'+str(i)][-1], c='dimgray', s=250-i*70, zorder=5)
        
        for i in range(self.num_enemy):
            if self.enemy_flags[i]:
                plt.scatter(self.enemy_points['x'+str(i)][-1], self.enemy_points['y'+str(i)][-1], s=250-i*50, marker='s', 
                            c=colors_enemy[i], zorder=10)
            else:
                plt.scatter(self.enemy_points['x'+str(i)][-1], self.enemy_points['y'+str(i)][-1], s=250-i*50, marker='s', c='dimgray', zorder=5)
        
        total_step_text = f'Total Step: {self.stepi}'
        other_text = (
            '\n\n'
            + 'Circle Survival: {}'.format(self.self_flags) + ', Square Survival: {}'.format(self.enemy_flags) + "\n"
            + 'Circle HP: [{}]'.format(', '.join(f"{blood:.2f}" for blood in self.self_bloods)) + ', Square HP: [{}]'.format(', '.join(f"{blood:.2f}" for blood in self.enemy_bloods))
        )
        
        plt.suptitle(other_text, fontsize=12, y=0.95)
        plt.subplots_adjust(top=0.8)

        plt.text(0.5, 1.16, total_step_text, fontsize=14, ha='center', va='bottom', transform=plt.gca().transAxes)

        plt.ylim(0, 100)
        plt.xlim(0, 200)
        plt.savefig('{}/{}_{}_samples.png'.format(path, n, step, dpi=300))

        frame = '{}/{}_{}_samples.png'.format(path, n, step)
        self.frame_files.append(frame)
        plt.close()

    def get_connected_positions(self, start_pos, network_adjacent_matrix, max_distance=2):
        connected = set()
        current_distance = 0
        current_layer = {start_pos}
        visited = {start_pos}
        
        while current_distance < max_distance and current_layer:
            next_layer = set()
            for pos in current_layer:
                connected.add(pos)
                neighbors = {i for i, is_adjacent in enumerate(network_adjacent_matrix[pos]) 
                           if is_adjacent == 1 and i not in visited}
                next_layer.update(neighbors)
                visited.update(neighbors)
            
            current_layer = next_layer
            current_distance += 1
        
        return list(connected)

    def select_close_positions(self, candidate_indexes, network_adjacent_matrix, num_positions, max_distance=2):
        if not candidate_indexes:
            return []
        
        candidate_indexes = set(candidate_indexes)
        
        while True:
            first_pos = random.choice(list(candidate_indexes))
            connected = self.get_connected_positions(first_pos, network_adjacent_matrix, max_distance)
            valid_candidates = set(connected) & candidate_indexes

            if len(valid_candidates) >= num_positions:
                positions = [first_pos]
                valid_candidates.remove(first_pos)
                positions.extend(random.sample(list(valid_candidates), num_positions - 1))
                return positions
