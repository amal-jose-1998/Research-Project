import gym
import torch
import time
import argparse
import numpy as np


def evaluate_policy(eval_env, agent_models, num_episodes=10):
    total_reward = {}
    for _ in range(num_episodes):
        done = False
        observations, _ = eval_env.reset()
        while not done:
            actions = {}
            for agent_name, model in zip(eval_env.agents, agent_models):
                state = torch.tensor(observations[agent_name], dtype=torch.float32)
                action = model.select_action(state, evaluate=True)
                actions[agent_name] = action

            next_observations, rewards, terminations, truncations, info = eval_env.step(actions)
            done = any(terminations.values()) or any(truncations.values())
            for agent_name in eval_env.agents:
                if agent_name not in total_reward:
                    total_reward[agent_name] = 0.0
                total_reward[agent_name] += rewards[agent_name]
            observations = next_observations
    observations, _ = eval_env.reset()
    for agent_name in eval_env.agents:
        total_reward[agent_name] = total_reward[agent_name] / num_episodes
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