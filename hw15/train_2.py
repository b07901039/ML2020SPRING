"""
Reinforce with baseline
save total_rewards and final_rewards at ./rewardNpy
save models at ./model
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

from model import PolicyGradientNetwork, PolicyGradientAgent, PolicyGradientNetwork_2, BaselineNetwork, BaselineAgent

if __name__ == "__main__":

    env = gym.make('LunarLander-v2')



   
    EPISODE_PER_BATCH = 10  # 每蒐集 5 個 episodes 更新一次 agent
    NUM_BATCH = 1000        # 總共更新 400 次

    gamma = 0.99
    momentum = 0
    lr = 1e-3
    msg = "baseline_adam2"
    # scheduler = {"step_size": 100, "gamma": 0.8}
    scheduler = None
    optimizer = "Adam" # "Adam" or "SGD"
    hidden_size = 16 # default: 16
    baseline_update_per_episode = 1
    baseline_hidden_size = 10

    # print("* 2nd")
    print("EPISODE_PER_BATCH:", EPISODE_PER_BATCH)
    print("NUM_BATCH:", NUM_BATCH)
    print("gamma:", gamma)
    print("msg:", msg)
    print("momentum:", momentum)
    print("lr:", lr)
    print("scheduler:", scheduler)
    print("optimizer:", optimizer)
    print("hidden:", hidden_size)
    print("baseline_update_per_episode:", baseline_update_per_episode)
    print("baseline_hidden_size:", baseline_hidden_size)

    print("------------------------------------------------")
    network = PolicyGradientNetwork(hidden_size)
    agent = PolicyGradientAgent(network, momentum, lr, scheduler, optimizer)
    BaselineNetwork = BaselineNetwork(baseline_hidden_size)
    BaselineAgent = BaselineAgent(BaselineNetwork)
    agent.network.train()  # 訓練前，先確保 network 處在 training 模式
    
    avg_total_rewards, avg_final_rewards = [], []

    # prg_bar = tqdm(range(NUM_BATCH))
    for batch in range(NUM_BATCH):

        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []
        mses = []

        # 蒐集訓練資料
        for episode in range(EPISODE_PER_BATCH):
            
            state = env.reset()
            total_reward, total_step = 0, 0

            per_step_rewards = [] 
            states = []
            while True:

                action, log_prob = agent.sample(state)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob)
                state = next_state
                states.append(state)
                total_reward += reward
                total_step += 1
                # print("total_step: ", total_step, end='\r')

                per_step_rewards.append(float(reward))

                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    # rewards.append(np.full(total_step, total_reward))  # 設定同一個 episode 每個 action 的 reward 都是 total reward
                    per_episode_rewards = [0 for i in range(total_step)]
                    for i in range(total_step):
                      for j in range(i, total_step):
                        per_episode_rewards[i] += gamma ** (j - i) * per_step_rewards[j]

                    for i in range(baseline_update_per_episode):
                        mses.append(BaselineAgent.learn(states, per_episode_rewards))
                        
                    
                    rewards.append(np.array(per_episode_rewards) - BaselineAgent.network(torch.Tensor(states).cuda()).cpu().detach().numpy())
                    ## rewards.append(np.array([sum(per_step_rewards[i:]) for i in range(total_step)]).reshape(-1, 1))
                    break

        # 紀錄訓練過程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        mseloss = np.mean(mses)
        # prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
        print("Epoch [{:d} / {:d}]: Total: {: 4.1f}, Final: {: 4.1f}, baseline loss: {:.4f}"
            .format(batch, NUM_BATCH, avg_total_reward, avg_final_reward, mseloss), end='\r')


        
        # 更新網路
        rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

    np.save(os.path.join("./rewardNpy", "total_reward_{}.npy".format(msg)), avg_total_rewards)
    np.save(os.path.join("./rewardNpy" ,"final_reward_{}.npy".format(msg)),avg_final_rewards)
    torch.save(agent.network.state_dict(), os.path.join("./model", "Model_{}.bin".format(msg)))
    torch.save(BaselineAgent.network.state_dict(), os.path.join("./model", "Model_baseline_{}.bin".format(msg)))
