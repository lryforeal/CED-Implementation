import torch
import ray
from model import PolicyNet, QNet
from worker import Worker
from parameter import *

class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network_self = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network_enemy = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_q_net = QNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network_self.to(self.device)
        self.local_network_enemy.to(self.device)
        self.local_q_net.to(self.device)

    def get_self_weights(self):
        return self.local_network_self.state_dict()

    def set_self_policy_net_weights(self, weights):
        self.local_network_self.load_state_dict(weights)

    def get_enemy_weights(self):
        return self.local_network_enemy.state_dict()

    def set_enemy_policy_net_weights(self, weights):
        self.local_network_enemy.load_state_dict(weights)

    def set_q_net_weights(self, weights1):
        self.local_q_net.load_state_dict(weights1)

    def do_job(self, episode_number):
        # save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        save_img = False
        base_checkpoint = torch.load(base_policy_path)
        self.local_network_self.load_state_dict(base_checkpoint['policy_model'])
        
        worker = Worker(self.meta_agent_id, self.local_network_self, self.local_network_enemy, self.local_q_net, episode_number, device=self.device, save_image=save_img, greedy=False, random_seed=RANDOM_SEED)
        worker.work(episode_number)

        job_results = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, perf_metrics

    def job(self, weights_set, episode_number):
        # print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        # self.set_self_policy_net_weights(weights_set[0])
        self.set_enemy_policy_net_weights(weights_set[1])
        self.set_q_net_weights(weights_set[2])

        job_results, metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return job_results, metrics, info

    def sample_job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_policy_net_weights(weights_set[0])
        self.set_q_net_weights(weights_set[1])

        # job_results, metrics = self.do_job(episode_number)
        # save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        save_img = False

        job_results = []
        for i in range(16):
            job_results.append([])
        
        worker = Worker(self.meta_agent_id, self.local_network, self.local_q_net, episode_number, device=self.device, save_image=save_img, greedy=False, random_seed=RANDOM_SEED)
        for i in range(128):
            worker.generate_batch_data()

        job_results = worker.episode_buffer
        # print("job_results: ", len(job_results[0]))
        # metrics = worker.perf_metrics

        # test = torch.stack(job_results[0]).to(self.device)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return job_results, info

    
  
@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):        
        super().__init__(meta_agent_id)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(1)
    out = ray.get(job_id)
    print(out[1])
