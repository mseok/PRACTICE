import argparse
import os
from pathlib import PosixPath
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from agent import ActorCritic, PolicyNetwork, ValueNetwork
from env import MoleculeEnv
from simulation import PPO
from losses import compute_ppo_loss, compute_value_loss
from utils import (
    get_abs_path,
    generate_dir,
    seed_all,
    to_device,
    init_logger,
    log_msg,
    log_arguments,
    seed_all,
)


def update_to_abs_path(FLAGS):
    update_dict = {}
    for key, value in vars(FLAGS).items():
        if type(value) == PosixPath:
            update_dict[key] = get_abs_path(value)
        else:
            update_dict[key] = value
    FLAGS_ = argparse.Namespace(**update_dict)
    return FLAGS_


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lamda", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=10001)
    parser.add_argument("--num_steps", type=int, default=101)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--trajectory_batch_size", type=int, default=50)
    parser.add_argument("--ppo_update_epoch", type=int, default=4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--clip_ratio", type=float, default=0.8)
    parser.add_argument("--entropy_ratio", type=float, default=0.01)
    parser.add_argument("--critic_ratio", type=float, default=1.0)
    parser.add_argument("--restart_fn", type=PosixPath)
    parser.add_argument("--log_fn", type=PosixPath, default="./logs/0")
    parser.add_argument("--save_dir", type=PosixPath, default="./save/0")
    parser.add_argument("--tensorboard_dir", type=PosixPath)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--natoms", type=int, default=3)
    parser.add_argument("--max_action", type=int, default=3)
    FLAGS, _ = parser.parse_known_args()
    FLAGS = update_to_abs_path(FLAGS)
    return FLAGS


class TrajectoryDataSet(Dataset):
    """
    Trajectory dataset from training
    """

    def __init__(self, traj: dict):
        self.traj = traj

    def __len__(self):
        try:
            return len(list(self.traj["states"]))
        except KeyError:
            return self.traj["values"].shape[0]

    def __getitem__(self, idx):
        return {key: values[idx] for key, values in self.traj.items()}


def update_ppo(FLAGS, traj_dataloader, model, optimizer, device=None):
    # results
    actor_losses = []
    ppo_losses = []
    entropy_losses = []
    value_losses = []
    total_losses = []

    for ppo_epoch in range(FLAGS.ppo_update_epoch):
        traj_iter = iter(traj_dataloader)
        batched_traj = next(traj_iter, None)
        end_of_episode = False
        while not end_of_episode:
            optimizer.zero_grad()
            if device is not None and device.type != "cpu":
                batched_traj = to_device(batched_traj, device)
            ob = batched_traj["states"]
            _, logprob = model.actor(*ob)

            logprob_old = batched_traj["logprobs"]
            # PPO loss
            ppo_loss, entropy_loss = compute_ppo_loss(
                logprob,
                batched_traj["logprobs"],
                batched_traj["advantages"],
                FLAGS.clip_ratio,
            )

            # Critic loss
            values_old = batched_traj["rwtogos"]
            values = model.critic(*ob)
            value_loss = compute_value_loss(values, values_old)

            # Gradient Ascent
            actor_loss = -ppo_loss - entropy_loss * FLAGS.entropy_ratio
            critic_loss = value_loss * FLAGS.critic_ratio

            total_loss = actor_loss + critic_loss

            # Update
            actor_loss.backward()
            critic_loss.backward()
            optimizer.step()

            actor_losses.append(actor_loss.data.cpu().numpy())
            ppo_losses.append(ppo_loss.data.cpu().numpy())
            entropy_losses.append(entropy_loss.data.cpu().numpy())
            value_losses.append(value_loss.data.cpu().numpy())
            total_losses.append(total_loss.data.cpu().numpy())

            batched_traj = next(traj_iter, None)
            if batched_traj is None:
                end_of_episode = True
    return {
        "ppo_loss": np.array(ppo_losses),
        "entropy_loss": np.array(entropy_losses),
        "value_loss": np.array(value_losses),
        "total_loss": np.array(total_losses),
    }


def main():
    FLAGS = parse_arguments()
    seed_all(FLAGS.seed)
    actor = PolicyNetwork()
    critic = ValueNetwork()
    model = ActorCritic(actor, critic)
    env = MoleculeEnv(FLAGS.natoms, FLAGS.max_action)

    log_dir = os.path.dirname(FLAGS.log_fn)
    generate_dir(log_dir)
    generate_dir(FLAGS.save_dir)
    enable_tensorboard = hasattr(FLAGS, "tensorboard_dir")
    if enable_tensorboard:
        writer = SummaryWriter(log_dir=FLAGS.tensorboard_dir)
    # Logger
    logger = init_logger(FLAGS.log_fn)
    log_arguments(logger, FLAGS)

    ppo_buffer = PPO(FLAGS.gamma, FLAGS.lamda, env, model)
    n_actor_param = sum(
        param.numel() for param in actor.parameters() if param.requires_grad
    )
    n_critic_param = sum(
        param.numel() for param in critic.parameters() if param.requires_grad
    )
    n_total_param = n_actor_param + n_critic_param
    log_msg(logger, f"Number of actor parameters: {n_actor_param}")
    log_msg(logger, f"Number of critic parameters: {n_critic_param}")
    log_msg(logger, f"Number of total parameters: {n_total_param}")

    if FLAGS.ngpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    msg = [
        "epoch",
        "steps",
        "ppo_loss",
        "entropy_loss",
        "value_loss",
        "total_loss",
        "time",
    ]
    msg = [m.ljust(12) for m in msg]
    log_msg(logger, "\t".join(msg))

    ppo_optimizer = optim.Adam(
        [
            {"params": actor.parameters(), "lr": FLAGS.lr_actor},
            {"params": critic.parameters(), "lr": FLAGS.lr_critic},
        ]
    )

    if device.type != "cpu":
        model.actor.to(device)
        model.critic.to(device)

    for epoch in range(FLAGS.epoch):
        trajectories = ppo_buffer.collect_trajectories(FLAGS.timesteps, device)

        for step in range(FLAGS.num_steps):
            st = time.time()
            # Prepare Trajectories for PPO update
            traj = next(trajectories, None)
            if traj is None:
                break
            traj_dataset = TrajectoryDataSet(traj)
            traj_dataloader = DataLoader(
                traj_dataset,
                batch_size=FLAGS.trajectory_batch_size,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                # collate_fn=mol_collate_fn,
            )
            losses = update_ppo(FLAGS, traj_dataloader, model, ppo_optimizer, device)
            ppo_keys, ppo_values = list(zip(*losses.items()))
            ppo_values = [ppo_value.mean() for ppo_value in ppo_values]
            et = time.time()

            msg = f"{epoch}\t{step}\t"
            ppo_values = [np.round(value, 3) for value in ppo_values]
            msg += "\t".join(list(map(str, ppo_values)))
            msg += f"\t{et - st:.3f}"
            msg = msg.split("\t")
            msg = [m.ljust(12) for m in msg]
            log_msg(logger, "\t".join(msg))

            if enable_tensorboard:
                total_step = epoch * (FLAGS.num_steps - 1) + step
                for idx, key in enumerate(ppo_keys):
                    writer.add_scalar(key, ppo_values[idx], total_step)

            if step % FLAGS.save_every == 0:
                fn = f"{epoch}_{step}.pt"
                torch.save(
                    actor.state_dict(), os.path.join(FLAGS.save_dir, "actor_" + fn)
                )
                torch.save(
                    critic.state_dict(), os.path.join(FLAGS.save_dir, "critic_" + fn)
                )
    return


if __name__ == "__main__":
    main()
