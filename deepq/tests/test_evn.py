import deepq.common.env_wrapper as evn

import gym
if __name__=="__main__":
    my_env = evn.EpisodicLifeEnv(gym.make("Breakout-v0"))
