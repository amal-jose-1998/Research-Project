import gym
import torch
import time
import argparse


def evaluate_policy(env, agents, render=False, turns = 10):
    average_score = 0
    for j in range(turns):
        s, infos = env.reset()
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.04)
            actions = {agent_name: agent.select_action(obs[agent_name], evaluate=True) for agent_name, agent in agents.items()}
            s_prime, r, done, info = env.step(a)
            ep_r += r
            steps += 1
            s = s_prime
        scores += ep_r
        print(ep_r)
    return int(scores/turns)


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


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)