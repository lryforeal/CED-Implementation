import ray
import numpy as np
import os
import torch

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def run_test():
    if not os.path.exists(trajectory_path):
        os.makedirs(trajectory_path)

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_network_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    if device == 'cuda':
        checkpoint_self = torch.load(f'{model_path}/enemy/checkpoint.pth')
        checkpoint_enemy = torch.load(f'{model_path}/no_move.pth')
        # checkpoint_self = torch.load(f'model/3V3-sp-both-offline/enemy/best_checkpoint_11800.pth')
        # checkpoint_enemy = torch.load(f'model/3V3-ced-0.01-zero-offline/enemy/best_checkpoint_11800.pth')
    else:
        checkpoint_self = torch.load(f'model/3V3-4.21-sp-beta_0/enemy/best_checkpoint_3000.pth', map_location = torch.device('cpu'))
        checkpoint_enemy = torch.load(f'{model_path}/no_move.pth', map_location = torch.device('cpu'))
        # checkpoint_self = torch.load(f'model/3V3-sp-both-offline/enemy/best_checkpoint_11800.pth', map_location = torch.device('cpu'))
        # checkpoint_enemy = torch.load(f'model/3V3-4.21-sp-beta_0/enemy/best_checkpoint_2500.pth', map_location = torch.device('cpu'))

    print('model path: ', model_path)
    print('model episode', checkpoint_self['episode'])
    global_network_self.load_state_dict(checkpoint_self['policy_model'])
    global_network_enemy.load_state_dict(checkpoint_enemy['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights_self = global_network_self.state_dict()
    weights_enemy = global_network_enemy.state_dict()
    # print(global_network.previous_downsample.state_dict())
    weights = [weights_self, weights_enemy]
    curr_test = 0

    step_history = []
    success_history = []
    fail_history = []
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(step_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                step_history.append(metrics['steps'])
                success_history.append(metrics['success_rate'])
                fail_history.append(metrics['loss_rate'])

            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total Test:', NUM_TEST)
        print('|#Average Steps:', np.array(step_history).mean())
        print('|#Average Win Rate:', np.array(success_history).mean())
        print('|#Average Fail Rate:', np.array(fail_history).mean())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network_self.to(self.device)
        self.local_network_enemy.to(self.device)

    def set_weights(self, weights):
        self.local_network_self.load_state_dict(weights[0])
        self.local_network_enemy.load_state_dict(weights[1])

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network_self, self.local_network_enemy, episode_number, device=self.device, save_image=SAVE_GIFS, greedy=False, random_seed=RANDOM_SEED)
        worker.work(episode_number)

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init(num_gpus=1)
    for i in range(NUM_RUN):
        run_test()
