from typing import List, Union, Optional

import numpy as np
import torch
from torch.distributions import Categorical


def compute_ppo_loss(logprobs, logprobs_old, advantages, clip_ratio):
    advantages = advantages.unsqueeze(1)
    device = advantages.device
    ppo_loss = torch.Tensor([0]).to(device)
    entropy = torch.Tensor([0]).to(device)
    for logprob, logprob_old in zip(logprobs, logprobs_old):
        ratio = torch.exp(logprob - logprob_old)
        clip = torch.clamp(ratio, min=1 - clip_ratio, max=1 + clip_ratio)
        ppo_loss += torch.min(ratio * advantages, clip * advantages).mean()

        # Entropy Term
        approx_kl = (logprob_old - logprob).mean()
        policy = Categorical(logprob)
        entropy += policy.entropy().mean()
    return ppo_loss, entropy


def compute_value_loss(values, values_old):
    return ((values - values_old) ** 2).mean()
