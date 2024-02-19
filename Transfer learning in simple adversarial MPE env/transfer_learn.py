from dddqn import dddQN_Agent,ReplayBuffer
import copy

def transfer_train(opt, agent_models):
    for a in range(0, opt.good_agents_pretrain+1):
        agent_models[a].load(f"dddQN_source_agent_{a}","simple_adversary_2")   # Load pretrained models for the first two good agents and the adversary
    for a in range(opt.good_agents_pretrain+1, opt.good_agents_transfer_train+1):
        agent_models[a] = dddQN_Agent(opt) # Create a new model for the third good agent