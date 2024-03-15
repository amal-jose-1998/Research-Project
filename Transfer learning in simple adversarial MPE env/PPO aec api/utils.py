import argparse
import wandb

def evaluate_policy(s, eval_env, agent_models, agent_id, agents, num_episodes=10):
    rewards = {}
    total_reward = {}
    for _ in range(num_episodes):
        eval_env.reset(seed = s+1)
        for agent_name in eval_env.agents:
            rewards[agent_name] = 0
        for agent in eval_env.agent_iter():
            id = agent_id[agent]
            observation, reward, termination, truncation, info = eval_env.last()
            if termination or truncation:
                a = None
            else:
                rewards[agent] += reward
                model = agent_models[id]
                a, p, v = model.choose_action(observation)
            eval_env.step(a)

        for agent_name in agents:
            if agent_name not in total_reward:
                total_reward[agent_name] = 0.0
            total_reward[agent_name] += rewards[agent_name]
    
    for agent_name in agents:
        total_reward[agent_name] = total_reward[agent_name] / num_episodes
        wandb.log({f'avg_reward on evaluation for {agent_name}': total_reward[agent_name]})
        print(f'avg_reward on evaluation for {agent_name}: ', total_reward[agent_name])
    return total_reward
                
def str2bool(v):
    '''Transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reset_coarse_tuner_counters(agents, coarse_tuner_counter):
    for s in agents:
        coarse_tuner_counter[s] = 0

def reset_episode_rewards(agents, total_game_reward):
    for s in agents:
        total_game_reward[s] = 0

def reset_iter_counters(agents, total_training_steps, total_steps):
    for s in agents:
        total_training_steps[s] = 0
        total_steps[s] = 0

def check_done_flag(termination, truncation):
    if termination or truncation:
        return True

def loop_iteration(opt, agents, agent_id, agent_models, env, eval_env, N):
    total_training_steps = {}
    total_steps = {} 
    total_game_reward = {}
    coarse_tuner_counter = {}
    if opt.transfer_train == True:
        reset_coarse_tuner_counters(agents, coarse_tuner_counter)
    # initialise the the counters for learning
    reset_iter_counters(agents, total_training_steps, total_steps)
    # initialise the the rewards for each agent
    reset_episode_rewards(agents, total_game_reward)
    for i in range(opt.games):
        print('episode:', i+1)
        wandb.log({'Episode': i+1})
        reset_episode_rewards(agents, total_game_reward)
        env.reset(seed=opt.seed)
        starting_point ={}
        next_state = {}
        current_state = {}
        action = {}
        prob = {}
        val = {}
        reward = {}
        done = {}
        for agent_name in env.agents:
            starting_point[agent_name] = True
            done[agent_name] = 0
        for agent in env.agent_iter():
            total_steps[agent]+=1
            wandb.log({f'total_steps {agent}': total_steps[agent]})
            observation, r, termination, truncation, info = env.last()
            id = agent_id[agent]      
            if not starting_point[agent]:
                next_state[agent] = observation
                reward[agent] = r
                wandb.log({f'rewards for {agent} at each step': r})
                total_game_reward[agent]+=r              
                if termination or truncation:
                    done[agent] = 1
                else:
                    done[agent] = 0
                model = agent_models[id]
                model.remember(current_state[agent], action[agent], prob[agent], val[agent], reward[agent], done[agent])
                starting_point[agent] = True        
            if termination or truncation:
                a = None
            else:
                model = agent_models[id]
                current_state[agent] = observation
                a, p, v = model.choose_action(observation)
                action[agent] = a
                prob[agent] = p
                val[agent] = v
                starting_point[agent] = False
            agnt = agent 
            env.step(a)    
            model = agent_models[id]
            if total_steps[agnt] % N == 0:
                if opt.transfer_train == True:
                    if coarse_tuner_counter[agent] < 50:
                        model.coarse_tuning_settings()
                        coarse_tuner_counter[agent]+=1
                        print("coarse tuning", coarse_tuner_counter[agent] ,"for", agent)
                    else:
                        model.fine_tuning_settings()
                model.learn(agnt)
                total_training_steps[agnt] += 1
                wandb.log({f'learning steps of {agnt}': total_training_steps[agnt]}) 
                if total_training_steps[agnt] % opt.save_interval == 0:
                    model.save_models()
                    print(agnt, ' model saved.')
                if total_training_steps[agnt] % opt.eval_interval == 0:
                    print("Evaluation")
                    score = evaluate_policy(opt.seed, eval_env, agent_models, agent_id, agents)
                    print('evaluation score at the training step: ',total_training_steps[agnt],f' of the {agnt}: ', score)    
        wandb.log({'episode rewards': total_game_reward})  
        print('episode rewards:', total_game_reward)   