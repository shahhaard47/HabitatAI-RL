import torch
import numpy as np

from .habitat import construct_envs

def make_vec_envs(args):
    env = construct_envs(args)
    env = VecPyTorch(env, args.device)
    print("returning torch envs")
    return env


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).int()
        return stg

    def close(self):
        return self.venv.close()

    # todo check if different device needed
    def get_gt_pose(self):
        return torch.tensor(self.venv.get_gt_pose()).float().to(self.device)

    def get_sim_pose(self):
        return torch.tensor(self.venv.get_sim_pose()).float().to(self.device)

    def get_gt_map(self):
        gt_map = self.venv.get_gt_map()
        return torch.from_numpy(gt_map).float().to(self.device)

    def get_sim_map(self):
        sim_map = self.venv.get_sim_map()
        return torch.from_numpy(sim_map).float().to(self.device)

    def get_goal_coords(self):
        return torch.tensor(self.venv.get_goal_coords()).int().to(self.device)
    
    def get_optimal_gt_action(self):
        action = self.venv.get_optimal_gt_action()
        return torch.tensor(action).int().to(self.device)

    def get_optimal_action(self):
        action = self.venv.get_optimal_action()
        return torch.tensor(action).int().to(self.device)