"""
load model at ./model
test TEST_NUM times and print out mean and std of total_rewards
"""
# %%capture
from pyvirtualdisplay import Display
# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()

# %matplotlib inline
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
# from tqdm.notebook import tqdm

# %%capture
import gym
import os

from model import PolicyGradientNetwork, PolicyGradientAgent, PolicyGradientNetwork_2

if __name__ == "__main__":
    
    msg="gamma0.9_adam"
    hidden_size = 16
    model_pth = "./model/Model_{}.bin".format(msg)
    TEST_NUM = 200

    env = gym.make('LunarLander-v2')
    network = PolicyGradientNetwork(hidden_size).cuda()
    agent = PolicyGradientAgent(network)

    agent.network.load_state_dict(torch.load(model_pth))
    agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式
    
    total_rewards = []
    for i in range(TEST_NUM):
      print("step: [{:d} / {:d}]".format(i, TEST_NUM), end = '\r')
      state = env.reset()

      # img = plt.imshow(env.render(mode='rgb_array'))

      total_reward = 0

      done = False
      while not done:
          action, _ = agent.sample(state)
          state, reward, done, _ = env.step(action)

          total_reward += reward

          # img.set_data(env.render(mode='rgb_array'))
          # display.display(plt.gcf())
          # display.clear_output(wait=True)
      total_rewards.append(total_reward)

    print("total_rewards, mean: {:.4f}, std: {:.4f}".format(np.mean(total_rewards), np.std(total_rewards)))