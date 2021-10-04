from copy import deepcopy
from typing import Optional

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.optim as optim

from agent import PolicyNetwork, ValueNetwork
from utils import to_device


def discounted_cumsum(array, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    n = array.shape[0]
    mul = np.ones((n, n)) * discount
    mul[np.tril_indices(n)] = 1
    mul = np.cumprod(mul, axis=1)
    mul = np.triu(mul)
    out = mul @ array
    return out


def detach_to_cpu(tensor):
    tensor_ = tensor.detach().cpu()
    return tensor_


class Buffer(object):
    def __init__(self):
        self.__init()

    def __init(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = np.empty(0)
        self.rewards = np.empty(0)
        return None

    def __del(self):
        del self.states, self.actions, self.logprobs, self.values, self.rewards
        if hasattr(self, "advantages"):
            del self.advantages
        if hasattr(self, "rwtogos"):
            del self.rwtogos

    def clear(self):
        self.__del()
        self.__init()

    def load(self, backup: dict):
        # Fill the last information
        for key, value in backup.items():
            setattr(self, key, value)
        return

    def backup_last(self):
        dic = {key: values[-1:] for key, values in self.__dict__.items()}
        return dic

    def cut_attributes(self):
        for key in self.__dict__.keys():
            if key == "advantages" or key == "rwtogos":
                continue
            setattr(self, key, self.__dict__[key][:-1])
        return


class PPO:
    def __init__(self, gamma, lamda, env, model):
        self.gamma = gamma
        self.lamda = lamda
        self.env = env
        self.model = model
        self.buffer = Buffer()
        self.timestep = 0

    def collect_trajectories(
        self, timesteps: int = 256, device: Optional[torch.device] = None
    ):
        while True:
            # Get observation about current generating mol
            ob = self.env.get_observation()
            if device is not None and device.type != "cpu":
                ob = to_device(ob, device)
            action, logprob = self.model.actor(*ob)
            action = list(map(detach_to_cpu, action))
            logprob = list(map(detach_to_cpu, logprob))
            action = list(map(lambda x: x.item(), action))
            self.buffer.states.append(self.detach_to_cpu_ob(ob))  # S0
            self.buffer.actions.append(action)  # A0
            self.buffer.logprobs.append(logprob)  # P0
            value = self.model.critic(*ob).squeeze().item()
            self.buffer.values = np.append(self.buffer.values, value)  # V0
            # Execute the action from the policy, get the rewards
            reward, done = self.env.step(action)
            if done:
                # print(Chem.MolToSmiles(self.env.mol))
                self.env.initialize()
            self.buffer.rewards = np.append(self.buffer.rewards, reward)  # R0
            if self.timestep > 0 and self.timestep % timesteps == 0:
                backup = self.buffer.backup_last()
                self.compute_rewards_to_go()
                self.compute_advantages()
                self.buffer.cut_attributes()
                buffer_dict = deepcopy(self.buffer.__dict__)
                self.buffer.clear()
                self.buffer.load(backup)
                self.timestep += 1
                yield buffer_dict
            self.timestep += 1

    def compute_rewards_to_go(self) -> None:
        """Compute the rewards to go after collecting trajectories"""
        rwtogo = discounted_cumsum(self.buffer.rewards, self.gamma)
        self.buffer.rwtogos = rwtogo[:-1]
        return

    def compute_advantages(self) -> None:
        """Compute the generalized advantage estimator GAE"""
        deltas = (
            self.buffer.rewards[:-1]
            + self.buffer.values[1:] * self.gamma
            - self.buffer.values[:-1]
        )
        factor = self.gamma * self.lamda
        advantage = discounted_cumsum(deltas, factor)
        self.buffer.advantages = advantage
        return

    def detach_to_cpu_ob(self, ob):
        detached_ob = []
        for tensor in ob:
            detached_ob.append(detach_to_cpu(tensor))
        return detached_ob

    def test(self, timesteps: int = 256, device=None, save_fn=None):
        self.episode_count = 0
        while True:
            # Get observation about current generating mol
            ob = self.env.get_observation()
            if device is not None and device.type != "cpu":
                ob = to_device(ob, device)
            action, logprob = self.model.actor(*ob)
            action = list(map(detach_to_cpu, action))
            logprob = list(map(detach_to_cpu, logprob))
            action = list(map(lambda x: x.item(), action))
            self.buffer.states.append(self.detach_to_cpu_ob(ob))  # S0
            self.buffer.actions.append(action)  # A0
            self.buffer.logprobs.append(logprob)  # P0
            value = self.model.critic(*ob).squeeze().item()
            self.buffer.values = np.append(self.buffer.values, value)  # V0
            # Execute the action from the policy, get the rewards
            reward, done = self.env.step(action)
            if done:
                smiles = Chem.MolToSmiles(self.env.mol)
                with open(save_fn, "a") as f:
                    f.write(smiles + "\n")
                self.episode_count += 1
                self.env.initialize()
            self.buffer.rewards = np.append(self.buffer.rewards, reward)  # R0
            if self.timestep > 0 and self.timestep % timesteps == 0:
                backup = self.buffer.backup_last()
                self.compute_rewards_to_go()
                self.compute_advantages()
                self.buffer.cut_attributes()
                buffer_dict = deepcopy(self.buffer.__dict__)
                self.buffer.clear()
                self.buffer.load(backup)
                self.timestep += 1
                yield buffer_dict

    def test(self, timesteps, save_fn):
        episode_count = 0
        success = 0
        fail = 0
        while True:
            ob = self.env.get_observation()
            action, logprob = self.model.actor(*ob)
            action = list(map(detach_to_cpu, action))
            logprob = list(map(detach_to_cpu, logprob))
            action = list(map(lambda x: x.item(), action))
            self.buffer.states.append(ob)  # S0
            self.buffer.actions.append(action)  # A0
            self.buffer.logprobs.append(logprob)  # P0
            value = self.model.critic(*ob).squeeze().item()
            self.buffer.values = np.append(self.buffer.values, value)  # V0
            # Execute the action from the policy, get the rewards
            reward, done = self.env.step(action)
            self.buffer.rewards = np.append(self.buffer.rewards, reward)  # R0
            if done:
                smiles = Chem.MolToSmiles(self.env.mol)
                episode_count += 1
                if smiles == "C1CC1":
                    success += 1
                else:
                    fail += 1
                with open(save_fn, "a") as f:
                    f.write(smiles + "\n")
                self.env.initialize()
                if episode_count == timesteps:
                    break
        backup = self.buffer.backup_last()
        self.compute_rewards_to_go()
        self.compute_advantages()
        self.buffer.cut_attributes()
        buffer_dict = deepcopy(self.buffer.__dict__)
        self.buffer.clear()
        self.buffer.load(backup)
        return buffer_dict, success, fail
