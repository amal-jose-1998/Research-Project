import numpy as np
import torch
import gym
from dddqn import dddQN_Agent,ReplayBuffer
import os, shutil
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule
from pettingzoo.mpe import simple_adversary_v3
import copy

# cd "Tutorials\Tutorial 2 simple adversarial transfer learning"

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=1E6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1E5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=1e4, help='random steps befor trainning,5E4 in DQN Nature')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--obs_dim', type=int, default=10, help='observation dimension')
parser.add_argument('--buffersize', type=int, default=1e4, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=1E3, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=True, help='True: update target net hardly(copy)')
parser.add_argument('--action_dim', type=int, default=5, help='no of possible actions')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--hidden', type=int, default=200, help='number of units in Fully Connected layer')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--good_agents', type=int, default=2, help='model trainning frequency')
opt = parser.parse_args()
print(opt)

def main():
    num_games = 5
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    env = simple_adversary_v3.parallel_env(render_mode="human", N=opt.good_agents, max_cycles=25, continuous_actions=False)
    eval_env = simple_adversary_v3.parallel_env(render_mode="human", N=2, max_cycles=5, continuous_actions=False)
    print('  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '\n')
  
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
        writepath = 'runs/S{}_{}'.format(opt.seed,'dddQN') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    agent_models = [] # agent[0] is the red agent
    for agent_id in range(opt.good_agents+1):  
        agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
        if agent_id==0:
            agent_opt.obs_dim = 8
        else:
            agent_opt.obs_dim = 10
        model = dddQN_Agent(agent_opt)
        #explore noise linearly annealed from 1.0 to 0.02 within 200k steps
        schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise)
        model.exp_noise = opt.exp_noise
        agent_models.append(model)

    buffer = ReplayBuffer(opt.obs_dim,max_size=int(opt.buffersize))
    #begin to iterate
    total_steps = -1
    for i in range(num_games):
        # Create a dictionary to store actions for each agent
        actions = {}
        while total_steps < opt.Max_train_steps:
            done = False
            terminations = False
            truncations = False
            s, infos = env.reset(seed=opt.seed)
            while not done:
                if terminations or truncations:
                    done = True
                else:
                    for agent_name in env.agents:
                        numerical_part = int(agent_name.split("_")[-1])
                        model = agent_models[numerical_part]
                        a = model.select_action(torch.tensor(s[agent_name]), evaluate=False)
                        actions[agent_name] = a

                    s_prime, r, terminations, truncations, info = env.step(actions)
                    
                    for agent_name in env.agents:
                        s[agent_name]=torch.tensor(s[agent_name])
                        s_prime[agent_name]=torch.tensor(s_prime[agent_name])
                    
                    s = torch.stack([s[key] for key in env.agents], dim=0)
                    s_prime = torch.stack([s_prime[key] for key in env.agents], dim=0)


                    buffer.add(s, actions, r, s_prime, done)
                    s = s_prime
                    
                    # train, e-decay, log, save
                    if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                        total_steps += 1
                        if total_steps % opt.train_freq == 0: 
                            for agent_name in env.agents:
                                numerical_part = int(agent_name.split("_")[-1])
                                model = agent_models[numerical_part]
                                model.train(buffer)
                                #e-greedy decay
                                model.exp_noise = schedualer.value(total_steps)
                        #record & log
                        if total_steps % opt.eval_interval == 0:
                            score = evaluate_policy(eval_env, env.agents)
                            if opt.write:
                                writer.add_scalar('ep_r', score, global_step=total_steps)
                                writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                            print('seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))

                        '''save model'''
                        if total_steps % opt.save_interval == 0:
                            model.save("dddQN","simple_adversary",int(total_steps/1000))


        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()