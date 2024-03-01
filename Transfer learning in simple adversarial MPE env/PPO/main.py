import gym
import numpy as np
from ppo import Agent
import argparse
from utils import plot_learning_curve, str2bool

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use wandb to record the training')
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in training steps.')
parser.add_argument('--eval_interval', type=int, default=5, help='Model evaluating interval, in training steps.')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--alpha', type=float, default=0.99, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--obs_dim', type=int, default=10, help='observation dimension')
parser.add_argument('--action_dim', type=int, default=5, help='no of possible actions')
parser.add_argument('--good_agents_source', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_target', type=int, default=3, help='no of good agents for the target')
parser.add_argument('--pretrain', type=str2bool, default=False, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=False, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--games', type=int, default=100, help='no of episodes')
parser.add_argument('--train_all_agents', type=str2bool, default=False, help='to select if all the agents in the target task are to be loaded with pretrained models')
parser.add_argument('--best_good_agent', type=int, default=2, help='best model for the good agent')
opt = parser.parse_args()
print(opt)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)