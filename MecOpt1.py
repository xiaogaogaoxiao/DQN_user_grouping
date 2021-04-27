# This file build the MEC environment and returns the optimized energy value.


import numpy as np
from scipy.special import lambertw
import random


class MecEnv:
    def __init__(
            self,
            n_ue=6,  # number of users
            ue_per_gp=2,  # number of users per group
            b=2e6,  # Bandwidth Hz
            n0_db=-174,  # Noise Spectral Density dBm
            min_r=100,  # meter
            max_r=1000,  # meter
            pl_exp=3.76,  # meter
    ):
        self.N_UE = n_ue
        self.UE_per_gp = ue_per_gp
        self.B = b
        self.MIN_R = min_r
        self.MAX_R = max_r
        self.PL_exp = pl_exp
        self.N0_dB = n0_db
        self.N0 = 10 ** ((self.N0_dB - 30) / 10)
        self.N_SC = self._find_n_sc()  # number of subcarriers

    def _find_n_sc(self):
        if self.N_UE % self.UE_per_gp == 0:
            return int(self.N_UE / self.UE_per_gp)

    def generate_channel(self):
        raw_r = self.MIN_R + (self.MAX_R - self.MIN_R) * np.random.rand(self.N_UE)
        raw_theta = 2 * np.pi * np.random.rand(self.N_UE)
        raw_ue_x = raw_r * np.cos(raw_theta)
        raw_ue_y = raw_r * np.sin(raw_theta)
        ue_dis = np.sqrt(np.power(raw_ue_x, 2) + np.power(raw_ue_y, 2))
        a = np.clip(np.random.standard_normal(size=self.N_UE), -0.8, 0.8)
        b = np.clip(np.random.standard_normal(size=self.N_UE), -0.8, 0.8)
        raw_ch_rayleigh = np.sqrt(0.5) * a + 1j * np.sqrt(0.5) * b
        # print(raw_ch_rayleigh)
        raw_ch = np.power(ue_dis, -0.5 * self.PL_exp) * raw_ch_rayleigh
        # print(raw_ch)
        ue_ch = np.power(np.abs(raw_ch), 2) / (self.B * self.N0)
        return ue_ch


class OptiMec:
    u_id_m = (1, 1)
    u_id_n = (0, 1)
    pw_m = 0.5
    tau_m = 0.03
    tau_n = 0.05
    channel_list = []

    def __init__(
            self,
            pw_m,
            tau_m,
            tau_n,
            u_id_m,
            u_id_n,
            c_list,
            dt_len,  # Size of data packet
            k0=1e-28,  # Effective capacitance coefficient
            c0=1e3,  # Number of CPU cycles required per nat

    ):
        self.B = MecEnv().B
        self.k0 = k0
        self.c0 = c0
        self.data_len = dt_len
        self.pw_m = pw_m
        self.channel_list = c_list
        if tau_m < tau_n:
            self.tau_m = tau_m
            self.tau_n = tau_n + 0.03
            self.u_id_m = u_id_m
            self.u_id_n = u_id_n
        elif tau_m >= tau_n:
            self.tau_m = tau_n
            self.tau_n = tau_m + 0.03
            self.u_id_m = u_id_n
            self.u_id_n = u_id_m
        self.t_r = self.tau_n - self.tau_m

    def find_ue_ch(self, u_id):
        ch_list = self.channel_list
        return ch_list[u_id]

    def pw_al_csi_case_eq(self):  # case pm=pn
        z1 = (3 * self.k0 * self.B * np.power(self.c0 * self.data_len, 3) * self.find_ue_ch(self.u_id_n)) / (
                np.power(self.tau_n, 2) * self.data_len)
        z2 = self.data_len / (self.B * self.tau_n)
        opt_beta = 1 - (2 / z2) * lambertw(0.5 * np.power(z1, -0.5) * z2 * np.exp(0.5 * z2))
        opt_beta = abs(opt_beta)
        pw_n = np.power(self.find_ue_ch(self.u_id_n), -1) * (
                np.exp((opt_beta * self.data_len) / (self.B * self.tau_n)) - 1)
        pw_r = pw_n
        return pw_n, pw_r, opt_beta

    def pw_al_csi_case_neq(self):  # case pm~=pn
        zu = (self.tau_m / (self.tau_n - self.tau_m)) * np.log(
            self.pw_m * self.find_ue_ch(self.u_id_m) * np.power(np.exp(self.data_len / (self.B * self.tau_m)) - 1, -1))
        z1 = (3 * self.k0 * self.B * np.power(self.c0 * self.data_len, 3) * self.find_ue_ch(self.u_id_n) * np.exp(
            2 * zu)) / (np.power(self.tau_n, 2) * self.data_len)
        z2 = self.data_len / (self.B * (self.tau_n - self.tau_m))
        opt_beta = 1 - (2 / z2) * lambertw(0.5 * np.power(z1, -0.5) * z2 * np.exp(0.5 * z2))
        opt_beta = abs(opt_beta)
        pw_n = np.power(self.find_ue_ch(self.u_id_n), -1) * (self.pw_m * self.find_ue_ch(self.u_id_m) * np.power(
            np.exp(self.data_len / (self.B * self.tau_m) - 1), -1) - 1)
        pw_r = np.power(self.find_ue_ch(self.u_id_n), -1) * (
                np.exp((opt_beta * self.data_len) / (self.B * self.t_r) - (self.tau_m / self.t_r) * np.log(
                    self.pw_m * self.find_ue_ch(self.u_id_m) * np.power(
                        np.exp(self.data_len / (self.B * self.tau_m)) - 1, -1))) - 1)
        return pw_n, pw_r, opt_beta

    # for 1n=0:
    def pw_al_qos(self):
        az = self.pw_m * self.find_ue_ch(self.u_id_m) + 1
        z1 = (3 * self.k0 * self.B * np.power(self.c0, 3) * np.power(self.data_len, 2) * self.find_ue_ch(
            self.u_id_n) * np.exp((self.t_r * np.log(az)) / self.tau_n)) / (np.power(self.tau_n, 2) * az)
        z2 = self.data_len / (self.B * self.tau_n)
        opt_beta = 1 - (2 / z2) * lambertw(0.5 * np.power(z1, -0.5) * z2 * np.exp(0.5 * z2))
        opt_beta = abs(opt_beta)
        pw_n = np.power(self.find_ue_ch(self.u_id_n), -1) * az * (np.exp(
            (opt_beta * self.data_len - self.t_r * self.B * np.log(az)) / (self.B * self.tau_n)) - 1)
        pw_r = np.power(self.find_ue_ch(self.u_id_n), -1) * (az * np.exp(
            (opt_beta * self.data_len - self.t_r * self.B * np.log(az)) / (self.B * self.tau_n)) - 1)
        # pw_r = np.power(self.find_ue_ch(self.u_id_n), -1) * (np.power(az, self.tau_m / self.tau_n) * np.exp(
        #     (opt_beta * self.data_len) / (self.B * self.tau_n)) - 1)
        if 0 < self.pw_m * self.find_ue_ch(self.u_id_m) <= np.exp((opt_beta * self.data_len) / (self.B * self.t_r)) - 1:
            return pw_n, pw_r, opt_beta
        else:
            return 0

    def pw_al_choose_eq_neq(self):
        det_ue_m = self.pw_m * self.find_ue_ch(self.u_id_m)
        _, _, beta_eq = self.pw_al_csi_case_eq()
        _, _, beta_neq = self.pw_al_csi_case_neq()
        test_lw_bound = np.exp(self.data_len / (self.B * self.tau_m)) - 1
        if test_lw_bound < det_ue_m <= np.exp((beta_neq * self.data_len) / (self.B * self.tau_m)) * test_lw_bound:
            return self.pw_al_csi_case_neq()
        elif det_ue_m <= test_lw_bound:
            return 0
        elif det_ue_m >= np.exp((beta_eq * self.data_len) / (self.B * self.tau_n)) * test_lw_bound:
            return self.pw_al_csi_case_eq()

    def pw_al_choose_qos_csi(self):
        pw_csi = self.pw_al_choose_eq_neq()
        pw_qos = self.pw_al_qos()
        if pw_csi != 0 and pw_qos != 0:
            en_csi = self.tau_m * pw_csi[0] + self.t_r * pw_csi[1] + (
                    self.k0 * (self.c0 * (1 - pw_csi[2]) * self.data_len) ** 3) / (self.tau_n) ** 2
            en_qos = self.tau_m * pw_qos[0] + self.t_r * pw_qos[1] + (
                    self.k0 * (self.c0 * (1 - pw_qos[2]) * self.data_len) ** 3) / (self.tau_n) ** 2
            if en_csi <= en_qos:
                if en_csi <= 0:
                    print('csi')
                # print('csi',pw_csi)
                return en_csi
            elif en_csi > en_qos:
                if en_qos <= 0:
                    print('qos')
                # print('qos',pw_qos)
                return en_qos
        elif pw_csi == 0 and pw_qos != 0:
            en_qos = self.tau_m * pw_qos[0] + self.t_r * pw_qos[1] + (
                    self.k0 * (self.c0 * (1 - pw_qos[2]) * self.data_len) ** 3) / (self.tau_n) ** 2
            # print('qos1',pw_qos)
            if en_qos <= 0:
                print('qos')
            return en_qos
        elif pw_csi != 0 and pw_qos == 0:
            en_csi = self.tau_m * pw_csi[0] + self.t_r * pw_csi[1] + (
                    self.k0 * (self.c0 * (1 - pw_csi[2]) * self.data_len) ** 3) / (self.tau_n) ** 2
            if en_csi <= 0:
                print('csi')
            return en_csi
        else:
            # print('Infeasible!')
            return 0


if __name__ == "__main__":
    import scipy.io

    mec_env = MecEnv()
    min_delay = 0.15
    max_delay = 0.22
    N_UE = 6
    n_step = 500
    channel_list_gen = [mec_env.generate_channel() for i in range(n_step)]
    delay_list_gen = [np.random.rand(N_UE) * (max_delay - min_delay) + min_delay for i in range(n_step)]
    mdic = {"channels": channel_list_gen}
    scipy.io.savemat('channel_list.mat', mdic)
    mdic = {"delays": delay_list_gen}
    scipy.io.savemat('delay_list.mat', mdic)
