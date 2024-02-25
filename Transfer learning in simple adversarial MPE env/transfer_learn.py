from dddqn import dddQN_Agent,ReplayBuffer
import copy
import torch
import wandb
from utils import evaluate_policy

def transfer_train(opt, schedualer, env, eval_env):
    num_games = opt.games
    agent_models = [] # agent[0] is the adversary
    agent_buffers = []
    loss = {}
    terminations = {}
    truncations = {}

    for a in range(0, opt.good_agents_pretrain+1):
        agent_models[a].load(f"dddQN_source_agent_{a}","simple_adversary_2")   # Load pretrained models for the first two good agents and the adversary
        agent_buffers[a]= ReplayBuffer(opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
    for a in range(opt.good_agents_pretrain+1, opt.good_agents_transfer_train+1):
        agent_models[a] = dddQN_Agent(opt) # Create a new model for the third good agent
        agent_buffers[a]= ReplayBuffer(opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
    
    total_steps = 0
    total_training_steps = 1
    for i in range(num_games):
        print('episode:', i)
        actions={}
        done = False
        s, infos = env.reset(seed=opt.seed)
        for agent_name in env.agents:
            terminations[agent_name] = False
            truncations[agent_name] = False
        while not done:
            if any(terminations.values()) or any(truncations.values()):
                print('episode',i, 'terminated at', total_steps)
                done = 1
            else:
                total_steps += 1
                j = 0
                for agent_name in env.agents:
                    model = agent_models[j]
                    buffer = agent_buffers[j]
                    j+=1
                    a = model.select_action(torch.tensor(s[agent_name]), evaluate=False)
                    actions[agent_name]=a

                s_prime, r, terminations, truncations, info = env.step(actions)

                j = 0
                flag = 0
                for agent_name in env.agents:
                    current_state = torch.tensor(s[agent_name])
                    next_state = torch.tensor(s_prime[agent_name])
                    reward = torch.tensor(r[agent_name])
                    action = torch.tensor(actions[agent_name])
                    if terminations[agent_name] or truncations[agent_name]:
                        done = 1
                    buffer = agent_buffers[j]
                    buffer.add(current_state, action, reward, next_state, done)
                    flag = 0
                    if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                        flag = 1
                        if total_steps % opt.train_freq == 0: 
                            model = agent_models[j]
                            loss[j] = model.train(buffer)
                            if opt.write:
                                wandb.log({f'training Loss for agent{j}': loss[j].item()})
                            model.exp_noise = schedualer.value(total_training_steps) #e-greedy decay
                            print('episode: ',i,'training step: ',total_training_steps,'loss of agent ',j,': ',loss[j].item())
                    j+=1

                if flag:
                    wandb.log({'training step': total_training_steps})
                    if total_training_steps % opt.eval_interval == 0:
                        score = evaluate_policy(opt.seed, eval_env, agent_models)
                        if opt.write:
                            wandb.log({'evaluation_env  avg_reward': score, 'total steps': total_steps, 'episode': i})
                            print("Evaluation")
                            print('env seed:',opt.seed+1,'evaluation score at the training step: ',total_training_steps,': ', score)
                    if total_training_steps % opt.save_interval == 0:
                        for a in range(opt.good_agents_transfer_train+1):
                            model = agent_models[a]
                            model.save(f"dddQN_target_agent_{a}","simple_adversary_3")                    
                    total_training_steps+=1

                s = s_prime

    env.close()
    eval_env.close()