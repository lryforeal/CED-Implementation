import ray
import numpy as np
import os
import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *

def get_all_model_paths(models_dir):
    model_paths = glob.glob(os.path.join(models_dir, "enemy/best_checkpoint_*.pth"))
    model_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return model_paths

def test_model(model_path, num_tests=1000):
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_network_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    
    if device.type == 'cuda':
        checkpoint_self = torch.load(model_path)
        checkpoint_enemy = torch.load(f'{model_path.rsplit("/", 2)[0]}/no_move.pth')
    else:
        checkpoint_self = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint_enemy = torch.load(f'{model_path.rsplit("/", 2)[0]}/no_move.pth', map_location=torch.device('cpu'))

    model_episode = 'unknown'
    if 'episode' in checkpoint_self:
        model_episode = checkpoint_self['episode']
    elif '_' in os.path.basename(model_path):
        model_episode = os.path.basename(model_path).split('_')[-1].split('.')[0]
    
    print('Model Path:', model_path)
    print('Model Episode:', model_episode)
    
    global_network_self.load_state_dict(checkpoint_self['policy_model'])
    global_network_enemy.load_state_dict(checkpoint_enemy['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    weights_self = global_network_self.state_dict()
    weights_enemy = global_network_enemy.state_dict()
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
        while len(step_history) < num_tests:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                step_history.append(metrics['steps'])
                success_history.append(metrics['success_rate'])
                fail_history.append(metrics['loss_rate'])

            if curr_test < num_tests:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        avg_steps = np.array(step_history).mean()
        std_steps = np.array(step_history).std()
        avg_success = np.array(success_history).mean()
        avg_fail = np.array(fail_history).mean()
        
        print('|#Total Test:', num_tests)
        print(f'|#Average Steps: {avg_steps:.2f} ± {std_steps:.2f}')
        print(f'|#Average Win Rate: {avg_success:.4f}')
        print(f'|#Average Fail Rate: {avg_fail:.4f}')

        for a in meta_agents:
            ray.kill(a)
            
        return {
            'model': os.path.basename(model_path),
            'episode': model_episode,
            'avg_steps': avg_steps,
            'std_steps': std_steps,
            'success_rate': avg_success,
            'fail_rate': avg_fail
        }
        
    except Exception as e:
        for a in meta_agents:
            ray.kill(a)
        return None

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
        worker = TestWorker(self.meta_agent_id, self.local_network_self, self.local_network_enemy, 
                           episode_number, device=self.device, save_image=SAVE_GIFS, 
                           greedy=False, random_seed=RANDOM_SEED)
        worker.work(episode_number)
        return worker.perf_metrics

    def job(self, weights, episode_number):
        self.set_weights(weights)
        metrics = self.do_job(episode_number)
        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }
        return metrics, info

def plot_results(results, model_name):
    """Plot test results"""
    df = pd.DataFrame(results)
    df = df.sort_values('episode')
    
    if isinstance(df['episode'].iloc[0], str):
        df['episode'] = df['episode'].astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    color = 'tab:green'
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Success Rate')
    ax.plot(df['episode'], df['success_rate'], 'o-', color=color, label='Success Rate')
    ax.tick_params(axis='y')
    ax.set_ylim(0, 1.05)

    ax.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('curve/' + model_name + '/model_performance_comparison.png', dpi=300)
    print("Chart saved as 'model_performance_comparison.png'")
    plt.close()

def main():
    model_name = ''
    models_dir = 'model/' + model_name + '/'
    
    ray.init()
    
    model_paths = get_all_model_paths(models_dir)
    print(f"{len(model_paths)} to be tested.")
    
    if not model_paths:
        print(f"No model file in: {models_dir}")
        return
    
    results = []
    for model_path in model_paths:
        result = test_model(model_path)
        if result:
            results.append(result)
        else:
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('curve/' + model_name + '/model_performance_results.csv', index=False)
    print("Saving to 'model_performance_results.csv'")
    
    if results:
        plot_results(results, model_name)

    ray.shutdown()

if __name__ == '__main__':
    main()