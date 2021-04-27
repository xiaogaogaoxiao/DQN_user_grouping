from MecOpt1 import MecEnv
from MecOpt1 import OptiMec
from drl_net import dqn
import torch
import numpy as np
import random

import matplotlib.pyplot as plt
from gen import pairs
import scipy.io

# env settings:
mec_env = MecEnv()
N_UE = mec_env.N_UE
N_SC = mec_env.N_SC
N_UEPG = mec_env.UE_per_gp
m_pwr = 0.5
data_len = 2e6
UE_position = np.zeros((2, N_UE))
# loop settings:
n_episode = 100
n_step = 500
channel_list_temp = scipy.io.loadmat('channel_list.mat')
channel_list = channel_list_temp['channels']
delay_list_temp = scipy.io.loadmat('delay_list.mat')
delay_list = delay_list_temp['delays']
comb_list = [x for x in pairs(N_UE)]
state_space = np.ones(N_UE * 3)
state_space_next = np.ones(N_UE * 3)
state_dim = len(state_space)
action_dim = len(comb_list)

# DQN training:
# initialize:
train_net = dqn(state_dim, action_dim, 20000, 64, 10, 1e-2, 0.9, 0.7)
all_reward = []
all_eng = []
all_rnd_eng = []
all_exhaust = []
for i_episode in range(n_episode):
    reward = 0
    rnd_eng = 0
    energy = 0
    for i_time in range(n_step):
        ch_list = channel_list[i_time]
        dl_list = delay_list[i_time]
        normalized_ch = ch_list / np.amax(ch_list)
        state_space[N_UE:2 * N_UE] = normalized_ch[(np.array(state_space[0:N_UE])).astype(int)]
        state_space[-N_UE:] = dl_list[(np.array(state_space[0:N_UE])).astype(int)]
        group_list = state_space[0:N_UE].reshape((N_UEPG, N_SC))
        group_ch_list = ch_list[(np.array(state_space[0:N_UE])).astype(int)]
        group_delay_list = state_space[-N_UE:]
        action = train_net.choose_action(state_space)
        if i_time == n_step - 1:
            continue
        state_space_next[0:N_UE] = np.reshape(comb_list[action], -1)
        ch_list = channel_list[i_time + 1]
        normalized_ch = ch_list / np.amax(ch_list)
        state_space[N_UE:2 * N_UE] = normalized_ch[(np.array(state_space[0:N_UE])).astype(int)]
        state_space_next[N_UE:2 * N_UE] = normalized_ch[(np.array(state_space[0:N_UE])).astype(int)]
        dl_list = delay_list[i_time + 1]
        state_space_next[-N_UE:] = dl_list[(np.array(state_space[0:N_UE])).astype(int)]
        group_list = state_space[0:N_UE].reshape((N_UEPG, N_SC))
        group_ch_list = ch_list[(np.array(state_space[0:N_UE])).astype(int)]
        group_delay_list = state_space[-N_UE:]
        E_next = sum(
            [OptiMec(m_pwr, group_delay_list[int(group_list[0, i])], group_delay_list[int(group_list[1, i])],
                     int(group_list[0, i]),
                     int(group_list[1, i]), group_ch_list, data_len).pw_al_choose_qos_csi() for i in
             range(N_SC)])
        instant_eng_new = E_next
        instant_reward = -1 * E_next
        train_net.store_memory(state_space, action, instant_reward, state_space_next)
        train_net.learn()
        state_space = state_space_next.copy()
        reward += instant_reward

        rnd_pair = random.sample(range(N_UE), N_UE)
        rnd_list = np.reshape(rnd_pair, (N_UEPG, N_SC))
        rnd_ch_list = ch_list[(np.array(rnd_pair[0:N_UE])).astype(int)]
        rnd_delay_list = dl_list[(np.array(rnd_pair[0:N_UE])).astype(int)]
        E_rnd = sum(
            [OptiMec(m_pwr, rnd_delay_list[int(rnd_list[0, i])], rnd_delay_list[int(rnd_list[1, i])],
                     int(rnd_list[0, i]),
                     int(rnd_list[1, i]), rnd_ch_list, data_len).pw_al_choose_qos_csi() for i in
             range(N_SC)])
        rnd_eng_inst = E_rnd
        rnd_eng += rnd_eng_inst
        energy += instant_eng_new

    print('Ep: {} | reward: {:.3f} |'.format(i_episode, round((energy / n_step), 3), ))
    all_reward.append((reward / n_step))
    all_eng.append((energy / n_step))
    all_rnd_eng.append(rnd_eng / n_step)
    # all_exhaust.append(exhaust_eng)
mean_rnd_eng = sum(all_rnd_eng) / len(all_rnd_eng)
rnd_baseline = [mean_rnd_eng for i in range(n_episode)]
mdic = {"all_reward": all_eng, "episode": "Energy"}
scipy.io.savemat("Energy_consumption_episode.mat", mdic)
plt.figure(1)
plt.plot(all_eng, 'r*-', label='DQN: LR = 0.001')
plt.plot(rnd_baseline, 'k*-', label='random')
plt.xlabel('Episode')
plt.ylabel('Energy Consumption')
plt.legend()


plt.show()

