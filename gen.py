from MecOpt import MecEnv
import numpy as np
from itertools import combinations

mec_env = MecEnv()
N_UE = mec_env.N_UE
N_SC = mec_env.N_SC
N_UEPG = mec_env.UE_per_gp


def pairs(n):
    if n % 2 == 1 or n < 2:
        print("no solution")
        return
    if n == 2:
        yield [[0, 1]]
    else:
        Sn_2 = pairs(n - 2)
        for s in Sn_2:
            yield s + [[n - 2, n - 1]]
            for i in (range(int(n / 2) - 1)):
                sn = list(s)
                sn.remove(s[i])
                yield sn + [[s[i][0], n - 2], [s[i][1], n - 1]]
                yield sn + [[s[i][1], n - 2], [s[i][0], n - 1]]


if __name__ == "__main__":
    comb_list = [x for x in pairs(10)]
    print(len(comb_list))
