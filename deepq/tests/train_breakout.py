import gym
import deepq.common.env_wrapper as env_warpper
import deepq.network.deep_network as dqn

def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    # assert 'NoFrameskip' in env.spec.id
    env = env_warpper.EpisodicLifeEnv(env)
    env = env_warpper.NoopResetEnv(env, noop_max=30)
    # env = env_warpper.MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = env_warpper.FireResetEnv(env)
    env = env_warpper.ProcessFrame84(env)
    env = env_warpper.FrameStack(env, 4)
    env = env_warpper.ClippedRewardsWrapper(env)
    return env

def main():
    env = gym.make("Breakout-v0")
    env = wrap_dqn(env)
    model = dqn.DeepQNet(env)
    act = model.train(
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99
    )
    # act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    print(__name__)
    main()