import torch
import ray
import os
from tqdm import tqdm

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *
import warnings

warnings.filterwarnings('ignore')

ray.init()

def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_policy_net_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_policy_net_ref = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    base_checkpoint = torch.load(base_policy_path)
    global_policy_net_self.load_state_dict(base_checkpoint['policy_model'])
    global_policy_net_enemy.load_state_dict(base_checkpoint['policy_model'])
    global_policy_net_ref.load_state_dict(base_checkpoint['policy_model'])
    global_q_net1.load_state_dict(base_checkpoint['q_net1_model'])

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

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

    # launch the first job on each runner
    job_list = []
    curr_episode = 0
    buffer_filled = False
    step_history = []
    success_history = []
    fail_history = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))

    # initialize training replay buffer
    experience_buffer = []
    for i in range(29):
        experience_buffer.append([])    

    # collect data from worker and do training
    try:
        while True:
            with tqdm(total=REPLAY_SIZE, desc="Filling Buffer", unit="samples") as pbar:
                while True:
                    # wait for any job to be completed
                    done_id, job_list = ray.wait(job_list)
                    # get the results
                    done_jobs = ray.get(done_id)
                    
                    # save experience and metric
                    for job in done_jobs:
                        job_results, metrics, info = job
                        step_history.append(metrics['total_steps'])
                        success_history.append(metrics['success_rate'])
                        fail_history.append(metrics['loss_rate'])
                        
                        if not buffer_filled:
                            new_samples = len(job_results[0])
                            for i in range(len(experience_buffer)):
                                experience_buffer[i] += job_results[i]
                            

                            pbar.update(new_samples)
                            
                            if len(experience_buffer[0]) >= REPLAY_SIZE:
                                buffer_filled = True
                                
                                for i in range(len(experience_buffer)):
                                    experience_buffer[i] = experience_buffer[i][:REPLAY_SIZE]

                                buffer_save_path = 'experience_buffer.pt'
                                print(f"\nSaving buffer data in: {buffer_save_path}")
                                torch.save(experience_buffer, buffer_save_path)

                                # print('|#Average steps:', np.array(step_history).mean())
                                # print('|#Steps std:', np.array(step_history).std())
                                # print('|#Average success rate:', np.array(success_history).mean())
                                # print('|#Average fail rate:', np.array(fail_history).mean())

                                experience_buffer = []
                                if os.path.exists(buffer_save_path):
                                    print(f"Loading buffer data from: {buffer_save_path}")
                                    experience_buffer = torch.load(buffer_save_path)
                                    print(f"Buffer size: {len(experience_buffer[0])}")

                                return

                    # launch new task
                    curr_episode += 1
                    job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
                    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
