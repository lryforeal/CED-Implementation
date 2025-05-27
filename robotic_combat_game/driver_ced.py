import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import random

from model import PolicyNet, QNet
from test_worker import TestWorker
from parameter import *
import warnings

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ray.init(num_gpus=1)
# ray.init(num_gpus=1, local_mode=True)
# ray.init()

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_path + '/enemy'):
    os.makedirs(model_path + '/enemy')

def writeToTensorBoard(writer, tensorboardData, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    reward, value, policyLoss, klDiv, qValueLoss, entropy, policyGradNorm, qValueGradNorm, log_alpha, alphaLoss = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/KL Div', scalar_value=klDiv, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alphaLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=qValueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policyGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=qValueGradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)


def run_test(curr_episode):
    # save_img = True if curr_episode % 15 == 0 else False
    save_img = False
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_network_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint_self = torch.load(f'{model_path}/enemy/checkpoint.pth')
        checkpoint_enemy = torch.load(base_policy_path)
    else:
        checkpoint_self = torch.load(f'{model_path}/enemy/checkpoint.pth', map_location = torch.device('cpu'))
        checkpoint_enemy = torch.load(base_policy_path, map_location = torch.device('cpu'))

    global_network_self.load_state_dict(checkpoint_self['policy_model'])
    global_network_enemy.load_state_dict(checkpoint_enemy['policy_model'])
    # weights = global_network_self.state_dict()
    
    worker = TestWorker(meta_agent_id=None, policy_net_self=global_network_self, policy_net_enemy=global_network_enemy, global_step=curr_episode % 20, device=device, save_image=save_img, greedy=False)
    worker.work(curr_episode)
    perf_metrics = worker.perf_metrics
    steps = perf_metrics['steps']
    win = perf_metrics['success_rate']
    fail = perf_metrics['loss_rate']

    # print('win: ', win, 'fail: ', fail, 'steps: ', steps)
    return steps, win, fail

def write_test(writer, avg_steps_list, avg_success_rate_list, avg_fail_rate_list, curr_episode):
    avg_steps = np.array(avg_steps_list).mean()
    avg_success_rate = np.array(avg_success_rate_list).mean()
    avg_fail_rate = np.array(avg_fail_rate_list).mean()
    writer.add_scalar(tag='Test/Average steps', scalar_value=avg_steps, global_step=curr_episode)
    writer.add_scalar(tag='Test/Success rate', scalar_value=avg_success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Test/Loss rate', scalar_value=avg_fail_rate, global_step=curr_episode)

def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_policy_net_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_policy_net_ref = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)  # not trainable when loaded from checkpoint, manually tune it for now
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net_self.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # initialize decay (not use)
    policy_lr_decay = optim.lr_scheduler.StepLR(global_policy_optimizer, step_size=DECAY_STEP, gamma=0.96)
    q_net1_lr_decay = optim.lr_scheduler.StepLR(global_q_net1_optimizer,step_size=DECAY_STEP, gamma=0.96)
    q_net2_lr_decay = optim.lr_scheduler.StepLR(global_q_net2_optimizer,step_size=DECAY_STEP, gamma=0.96)
    log_alpha_lr_decay = optim.lr_scheduler.StepLR(log_alpha_optimizer, step_size=DECAY_STEP, gamma=0.96)
        
    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / (K_SIZE + N_ENEMY) ** N_SELF))

    curr_episode = 0
    target_q_update_counter = 1

    base_checkpoint = torch.load(base_policy_path)
    global_policy_net_self.load_state_dict(base_checkpoint['policy_model'])
    global_policy_net_enemy.load_state_dict(base_checkpoint['policy_model'])
    global_policy_net_ref.load_state_dict(base_checkpoint['policy_model'])
    global_q_net1.load_state_dict(base_checkpoint['q_net1_model'])
    global_q_net2.load_state_dict(base_checkpoint['q_net2_model'])

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_policy_net_self.load_state_dict(checkpoint['policy_model'])
        global_policy_net_enemy.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        policy_lr_decay.load_state_dict(checkpoint['policy_lr_decay'])
        q_net1_lr_decay.load_state_dict(checkpoint['q_net1_lr_decay'])
        q_net2_lr_decay.load_state_dict(checkpoint['q_net2_lr_decay'])
        log_alpha_lr_decay.load_state_dict(checkpoint['log_alpha_lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    
    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights_self = global_policy_net_self.to(local_device).state_dict()
        policy_weights_enemy = global_policy_net_enemy.to(local_device).state_dict()
        policy_weights_ref = global_policy_net_ref.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
        global_policy_net_self.to(device)
        global_policy_net_enemy.to(device)
        global_policy_net_ref.to(device)
        global_q_net1.to(device)
    else:
        policy_weights_self = global_policy_net_self.to(local_device).state_dict()
        policy_weights_enemy = global_policy_net_enemy.to(local_device).state_dict()
        policy_weights_ref = global_policy_net_ref.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
    weights_set.append(policy_weights_self)
    weights_set.append(policy_weights_enemy)
    weights_set.append(q_net1_weights)

    # distributed training if multiple GPUs available
    dp_policy_self = nn.DataParallel(global_policy_net_self)
    dp_policy_enemy = nn.DataParallel(global_policy_net_enemy)
    dp_policy_ref = nn.DataParallel(global_policy_net_ref)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # initialize metric collector
    training_data = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(29):
        experience_buffer.append([])
    
    experience_buffer = []
    buffer_save_path = 'experience_buffer.pt'
    if os.path.exists(buffer_save_path):
        print(f"Loading buffer data from: {buffer_save_path}")
        experience_buffer = torch.load(buffer_save_path)
        buffer_filled = True
    else:
        raise FileNotFoundError("No buffer data path found")

    # collect data from worker and do training
    try:
        maxreward = 0

        while True:
            # launch new task
            print('curr_episode: ', curr_episode)
            curr_episode += 1
            
            # start training
            newmaxreward = 0
            if curr_episode % 1 == 0 and buffer_filled:
                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for _ in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    self_node_inputs_batch = torch.stack(rollouts[0]).to(device)
                    self_all_edge_inputs_batch = torch.stack(rollouts[1]).to(device)
                    self_all_enemy_inputs_batch = torch.stack(rollouts[2]).to(device)
                    self_all_current_inputs_batch = torch.stack(rollouts[3]).to(device)
                    self_node_padding_mask_batch = torch.stack(rollouts[4]).to(device)
                    self_all_edge_padding_mask_batch = torch.stack(rollouts[5]).to(device)
                    self_edge_mask_batch = torch.stack(rollouts[6]).to(device)
                    self_all_enemy_padding_mask_batch = torch.stack(rollouts[7]).to(device)
                    enemy_node_inputs_batch = torch.stack(rollouts[8]).to(device)
                    enemy_all_edge_inputs_batch = torch.stack(rollouts[9]).to(device)
                    enemy_all_enemy_inputs_batch = torch.stack(rollouts[10]).to(device)
                    enemy_all_current_inputs_batch = torch.stack(rollouts[11]).to(device)
                    enemy_node_padding_mask_batch = torch.stack(rollouts[12]).to(device)
                    enemy_all_edge_padding_mask_batch = torch.stack(rollouts[13]).to(device)
                    enemy_edge_mask_batch = torch.stack(rollouts[14]).to(device)
                    enemy_all_enemy_padding_mask_batch = torch.stack(rollouts[15]).to(device)
                    self_action_batch = torch.stack(rollouts[16]).to(device)
                    enemy_action_batch = torch.stack(rollouts[17]).to(device)
                    enemy_base_logp_batch = torch.stack(rollouts[18]).to(device)
                    reward_batch = torch.stack(rollouts[19]).to(device)
                    done_batch = torch.stack(rollouts[20]).to(device)
                    next_node_inputs_batch = torch.stack(rollouts[21]).to(device)
                    next_all_edge_inputs_batch = torch.stack(rollouts[22]).to(device)
                    next_all_enemy_inputs_batch = torch.stack(rollouts[23]).to(device)
                    next_all_current_inputs_batch = torch.stack(rollouts[24]).to(device)
                    next_node_padding_mask_batch = torch.stack(rollouts[25]).to(device)
                    next_all_edge_padding_mask_batch = torch.stack(rollouts[26]).to(device)
                    next_edge_mask_batch = torch.stack(rollouts[27]).to(device)
                    next_all_enemy_padding_mask_batch = torch.stack(rollouts[28]).to(device)
                    q_net_node_inputs_batch = self_node_inputs_batch
                    q_net_next_node_inputs_batch = next_node_inputs_batch
                    
                    action_list = []
                    for i in range(BATCH_SIZE):
                        action = 0
                        for j in range(N_SELF):
                            action += self_action_batch[i, j] * (K_SIZE + N_ENEMY) ** j
                        action_list.append(action)

                    # action_batch = action_batch.unsqueeze(1)
                    self_action_batch = torch.tensor(action_list).unsqueeze(1).unsqueeze(1).to(device) # (batch_size, 1, 1)
                    self_action_batch = self_action_batch.to(torch.int64)

                    action_list = []
                    for i in range(BATCH_SIZE):
                        action = 0
                        for j in range(N_ENEMY):
                            action += enemy_action_batch[i, j] * (K_SIZE + N_SELF) ** j
                        action_list.append(action)
                    enemy_action_batch = torch.tensor(action_list).unsqueeze(1).unsqueeze(1).to(device) # (batch_size, 1, 1)
                    enemy_action_batch = enemy_action_batch.to(torch.int64)

                    # SAC
                    with torch.no_grad():
                        q_values1, _ = dp_q_net1(q_net_node_inputs_batch, self_all_edge_inputs_batch, self_all_enemy_inputs_batch, self_all_current_inputs_batch, self_node_padding_mask_batch, self_all_edge_padding_mask_batch, self_edge_mask_batch)
                        q_values2, _ = dp_q_net2(q_net_node_inputs_batch, self_all_edge_inputs_batch, self_all_enemy_inputs_batch, self_all_current_inputs_batch, self_node_padding_mask_batch, self_all_edge_padding_mask_batch, self_edge_mask_batch)
                        q_values = torch.min(q_values1, q_values2)
                        
                        logps_ref_list = []
                        for i in range(N_SELF):
                            logs_ref = dp_policy_ref(self_node_inputs_batch, self_all_edge_inputs_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], self_all_enemy_inputs_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY], \
                                                                self_all_current_inputs_batch[:, :, i].unsqueeze(2), self_node_padding_mask_batch, self_all_edge_padding_mask_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], \
                                                                self_edge_mask_batch, self_all_enemy_padding_mask_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY])
                            logps_ref_list.append(logs_ref)
                        logps_ref = torch.stack(logps_ref_list, dim=0)
                        logp = torch.zeros_like(q_values).to(device)
                        
                        tensor_0 = logps_ref[0, :, :].unsqueeze(2).unsqueeze(3)  # (batch_size, K_SIZE, 1, 1)
                        tensor_1 = logps_ref[1, :, :].unsqueeze(1).unsqueeze(3)  # (batch_size, 1, K_SIZE, 1)
                        tensor_2 = logps_ref[2, :, :].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, K_SIZE)
                        result = tensor_0 + tensor_1 + tensor_2
                        logp_ref = result.permute(0, 3, 2, 1).reshape(BATCH_SIZE, (K_SIZE+N_ENEMY)**N_SELF, -1)  # (batch_size, K_SIZE**N_SELF, 1)

                    logps_list = []
                    policy_loss = 0
                    for i in range(N_SELF):
                        logps_list.append(dp_policy_self(self_node_inputs_batch, self_all_edge_inputs_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], self_all_enemy_inputs_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY], \
                                                         self_all_current_inputs_batch[:, :, i].unsqueeze(2), self_node_padding_mask_batch, self_all_edge_padding_mask_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], \
                                                         self_edge_mask_batch, self_all_enemy_padding_mask_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY]))
                    logps = torch.stack(logps_list, dim=0)
                    
                    logp = torch.zeros_like(q_values).to(device)
                    tensor_0 = logps[0, :, :].unsqueeze(2).unsqueeze(3)  # (batch_size, K_SIZE, 1, 1)
                    tensor_1 = logps[1, :, :].unsqueeze(1).unsqueeze(3)  # (batch_size, 1, K_SIZE, 1)
                    tensor_2 = logps[2, :, :].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, K_SIZE)
                    result = tensor_0 + tensor_1 + tensor_2
                    logp = result.permute(0, 3, 2, 1).reshape(BATCH_SIZE, (K_SIZE+N_ENEMY)**N_SELF, -1)  # (batch_size, K_SIZE**N_SELF, 1)
                    
                    kl_div = ((logp-logp_ref) * logp.exp()).squeeze(2).sum(dim=-1).mean()
                    policy_loss = torch.sum((logp.exp() * (log_alpha.exp().detach() * logp - q_values.detach())), dim=1).mean()
                    policy_loss += BETA * kl_div
                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net_self.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()

                    with torch.no_grad():
                        next_q_values1, _ = dp_target_q_net1(q_net_next_node_inputs_batch, next_all_edge_inputs_batch, next_all_enemy_inputs_batch, next_all_current_inputs_batch, next_node_padding_mask_batch, next_all_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values2, _ = dp_target_q_net2(q_net_next_node_inputs_batch, next_all_edge_inputs_batch, next_all_enemy_inputs_batch, next_all_current_inputs_batch, next_node_padding_mask_batch, next_all_edge_padding_mask_batch, next_edge_mask_batch)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        
                        next_logps_list = []
                        for i in range(N_SELF):
                            next_logps_list.append(dp_policy_self(next_node_inputs_batch, next_all_edge_inputs_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], next_all_enemy_inputs_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY], next_all_current_inputs_batch[:, :, i].unsqueeze(2), self_node_padding_mask_batch, next_all_edge_padding_mask_batch[:, :, i*K_SIZE:(i+1)*K_SIZE], self_edge_mask_batch, next_all_enemy_padding_mask_batch[:, :, i*N_ENEMY:(i+1)*N_ENEMY]))

                        next_logps = torch.stack(next_logps_list, dim=0)
                        next_logp = torch.zeros_like(next_q_values).to(device)

                        tensor_0 = next_logps[0, :, :].unsqueeze(2).unsqueeze(3)  # (batch_size, K_SIZE, 1, 1)
                        tensor_1 = next_logps[1, :, :].unsqueeze(1).unsqueeze(3)  # (batch_size, 1, K_SIZE, 1)
                        tensor_2 = next_logps[2, :, :].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, K_SIZE)
                        result = tensor_0 + tensor_1 + tensor_2
                        next_logp = result.permute(0, 3, 2, 1).reshape(BATCH_SIZE, (K_SIZE+N_ENEMY)**N_SELF, -1)  # (batch_size, K_SIZE**N_SELF, 1)

                        value_prime_batch = torch.sum(next_logp.exp() * (next_q_values - log_alpha.exp() * next_logp), dim=1).unsqueeze(1)
                        target_q_batch = reward_batch + GAMMA * (1 - done_batch) * value_prime_batch 

                    mse_loss = nn.MSELoss()
                    q_values1, _ = dp_q_net1(q_net_node_inputs_batch, self_all_edge_inputs_batch, self_all_enemy_inputs_batch, self_all_current_inputs_batch, self_node_padding_mask_batch, self_all_edge_padding_mask_batch, self_edge_mask_batch)
                    # print('q_values1: ', q_values1.shape, 'action_batch: ', action_batch.shape)
                    q1 = torch.gather(q_values1, 1, self_action_batch)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000, norm_type=2)
                    global_q_net1_optimizer.step()
                    
                    q_values2, _ = dp_q_net2(q_net_node_inputs_batch, self_all_edge_inputs_batch, self_all_enemy_inputs_batch, self_all_current_inputs_batch, self_node_padding_mask_batch, self_all_edge_padding_mask_batch, self_edge_mask_batch)
                    q2 = torch.gather(q_values2, 1, self_action_batch)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000, norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).squeeze(2).sum(dim=-1).mean()
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()
                    
                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    
                    target_q_update_counter += 1


                # data record to be written in tensorboard
                data = [reward_batch.mean().item(), value_prime_batch.mean().item(), policy_loss.item(), kl_div.mean().item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(), alpha_loss.item()]
                training_data.append(data)
                newmaxreward = max(maxreward, data[0])

                # write record to tensorboard
                if len(training_data) >= SUMMARY_WINDOW and curr_episode % 15 == 0:
                    writeToTensorBoard(writer, training_data, curr_episode)
                    training_data = []

                # get the updated global weights
                weights_set = []
                if device != local_device:
                    policy_weights_self = global_policy_net_self.to(local_device).state_dict()
                    q_net1_weights = global_q_net1.to(local_device).state_dict()
                    global_policy_net_self.to(device)
                    global_q_net1.to(device)
                else:
                    policy_weights_self = global_policy_net_self.to(local_device).state_dict()
                    q_net1_weights = global_q_net1.to(local_device).state_dict()

                # update enemy policy
                if curr_episode % 100 == 0:
                    global_policy_net_enemy.load_state_dict(policy_weights_self)
                    print('update enemy policy')
                    if device != local_device:
                        policy_weights_enemy = global_policy_net_enemy.to(local_device).state_dict()
                        global_policy_net_enemy.to(device)
                    else:
                        policy_weights_enemy = global_policy_net_enemy.to(local_device).state_dict()

                    print('Saving model', end='\n')
                    checkpoint = {"policy_model": global_policy_net_enemy.state_dict(),
                                    "episode": curr_episode,
                            }

                    path_checkpoint = "./" + model_path + "/enemy/best_checkpoint_{}.pth".format(curr_episode)
                    torch.save(checkpoint, path_checkpoint)
                    path_checkpoint = "./" + model_path + "/enemy/checkpoint.pth"
                    torch.save(checkpoint, path_checkpoint)

                    print('Saved model', end='\n')
                
                weights_set.append(policy_weights_self)
                weights_set.append(policy_weights_enemy)
                weights_set.append(q_net1_weights)
                
                # update the target q net
                if target_q_update_counter > 64:
                    print("update target q net")
                    target_q_update_counter = 1
                    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                    global_target_q_net1.eval()
                    global_target_q_net2.eval()

                # save the model
                if curr_episode % 32 == 0 or newmaxreward >= maxreward:
                    print('Saving model', end='\n')
                    checkpoint = {"policy_model": global_policy_net_self.state_dict(),
                                    "q_net1_model": global_q_net1.state_dict(),
                                    "q_net2_model": global_q_net2.state_dict(),
                                    "log_alpha": log_alpha,
                                    "policy_optimizer": global_policy_optimizer.state_dict(),
                                    "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                                    "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                                    "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                                    "episode": curr_episode,
                                    "policy_lr_decay": policy_lr_decay.state_dict(),
                                    "q_net1_lr_decay": q_net1_lr_decay.state_dict(),
                                    "q_net2_lr_decay": q_net2_lr_decay.state_dict(),
                                    "log_alpha_lr_decay": log_alpha_lr_decay.state_dict()
                            }
                    path_checkpoint = "./" + model_path + "/checkpoint.pth"
                    torch.save(checkpoint, path_checkpoint)
                    if newmaxreward > maxreward:
                        path_checkpoint = "./" + model_path + "/best_checkpoint_{}.pth".format(curr_episode)
                        torch.save(checkpoint, path_checkpoint)
                    maxreward = newmaxreward
                    print('Saved model', end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Stop training.")


if __name__ == "__main__":
    main()
