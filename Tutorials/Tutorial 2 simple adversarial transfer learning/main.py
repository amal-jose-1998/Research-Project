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
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=10, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=50, help=' min no of replay buffer experiences to start training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--obs_dim', type=int, default=10, help='observation dimension')
parser.add_argument('--buffersize', type=int, default=1e5, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=10, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=True, help='True: update target net hardly(copy)')
parser.add_argument('--action_dim', type=int, default=5, help='no of possible actions')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--hidden', type=int, default=200, help='number of units in Fully Connected layer')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--good_agents', type=int, default=2, help='no of good agents')
parser.add_argument('--games', type=int, default=100, help='no of episodes')
opt = parser.parse_args()
print(opt)

def main():
    num_games = opt.games
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    env = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents, max_cycles=100, continuous_actions=False)
    eval_env = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents, max_cycles=100, continuous_actions=False)
  
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
        writepath = 'runs/S{}_{}'.format(opt.seed,'dddQN') + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    agent_models = [] # agent[0] is the adversary
    agent_buffers = []
    for agent_id in range(opt.good_agents+1):  
        agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
        if agent_id==0:
            agent_opt.obs_dim = 8
        else:
            agent_opt.obs_dim = 10
        model = dddQN_Agent(agent_opt) # Create a model for each agent
        schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #explore noise linearly annealed from 1.0 to 0.02 within 200k steps
        model.exp_noise = opt.exp_noise
        agent_models.append(model)
        buffer = ReplayBuffer(agent_opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
        agent_buffers.append(buffer)
    e = 0
    for j in range(num_games):
        print('episode:', j)
        actions={}
        done = False
        terminations = False
        truncations = False
        s, infos = env.reset()
        total_steps = 0
        while not done:
            if terminations or truncations:
                done = 1
            else:
                e += 1
                i = 0
                for agent_name in env.agents:
                    model = agent_models[i]
                    buffer = agent_buffers[i]
                    i+=1
                    a = model.select_action(torch.tensor(s[agent_name]), evaluate=False)
                    actions[agent_name]=a
                s_prime, r, terminations, truncations, info = env.step(actions)
                i=0

                for agent_name in env.agents:
                    current_state = torch.tensor(s[agent_name])
                    next_state = torch.tensor(s_prime[agent_name])
                    reward = torch.tensor(r[agent_name])
                    action = torch.tensor(actions[agent_name])
                    if terminations[agent_name] or truncations[agent_name]:
                        done = 1
                    buffer = agent_buffers[i]
                    buffer.add(current_state, action, reward, next_state, done)
                    i+=1
                    # train, e-decay, log, save
                    if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                        total_steps += 1
                        if total_steps % opt.train_freq == 0: 
                            for a in range(opt.good_agents+1):
                                model = agent_models[a]
                                loss = model.train(buffer)
                                #e-greedy decay
                                model.exp_noise = schedualer.value(total_steps)
                                print('loss',total_steps,': ',loss)
                s = s_prime
                        #record & log
                if e % opt.eval_interval == 0:
                    score = evaluate_policy(eval_env, agent_models )
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=e)
                        writer.add_scalar('noise', model.exp_noise, global_step=e)
                        print('seed:',opt.seed,'steps: {}k'.format(int(e/1000)),'score:', int(score))

                        '''save model'''
                if e % opt.save_interval == 0:
                    model.save("dddQN","simple_adversary",int(e/1000))


    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()