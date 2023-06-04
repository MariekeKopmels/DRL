import time
import numpy as np

import gym

from stable_baselines3 import PPO

def run_environment():
    return 1


if __name__ == '__main__':
    timestamp = time.time()
    results = run_environment()
    log_number = 0
    np.save(f"group_56_part2_rewards_{log_number}.npy", results)
    print("Done in {:.3f} seconds".format(time.time()-timestamp))

