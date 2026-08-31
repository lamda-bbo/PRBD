import numpy as np
import gymnasium as gym
from benchmark.filter import RunningStat


ENV_NAME = ['HalfCheetah-v2', 'Walker2d-v4', 'Hopper-v4']

class GymV2Wrapper:
    """
    将原本 v2 接口适配到 v4/v5:
    - env.reset() 返回 obs
    - env.step(action) 返回 obs, reward, done, info
    """
    def __init__(self, env_name, seed=None):
        self.env = gym.make(env_name)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, r, done, info

    def __getattr__(self, name):
        # 将其他方法直接透传，比如 observation_space, action_space
        return getattr(self.env, name)


class RLEnv:
    def __init__(self, env_name=ENV_NAME[0], seed=2021):
        self.env_name = env_name
        self.env = GymV2Wrapper(self.env_name, seed=seed)
        state_dims = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        
        self.dims = state_dims * action_dims
        self.policy_shape = (action_dims, state_dims)
        self.lb = -1 * np.ones(self.dims)
        self.ub = 1 * np.ones(self.dims)
        self.rs = RunningStat(state_dims)
        
        self.num_rollouts = 3
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        M = x.reshape(self.policy_shape)
        total_r = 0
        n_samples = 0
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            while True:
                self.rs.push(obs)
                norm_obs = (obs - self.rs.mean) / (self.rs.std + 1e-6)
                action = np.dot(M, norm_obs)
                obs, r, done, _ = self.env.step(action)
                total_r += r
                n_samples += 1
                if done:
                    break
        
        return total_r / self.num_rollouts, n_samples / 3
    

if __name__ == '__main__':
    f = RLEnv()
    