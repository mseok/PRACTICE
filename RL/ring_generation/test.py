import glob
from multiprocessing import Pool
import os
import sys

import numpy as np
import torch
import rdkit
from rdkit import Chem

from agent import ActorCritic, PolicyNetwork, ValueNetwork
from env import MoleculeEnv
from simulation import PPO
from utils import seed_all, generate_dir


def succ_len(fn):
    arr = np.load(fn)
    arr_ = arr[arr > 1]
    return arr_.shape[0]


def test(fns=None, count=1000, exp=None):
    if fns:
        actor_fn, critic_fn = fns
        if not os.path.exists(actor_fn) or not os.path.exists(critic_fn):
            raise FileNotFoundError(actor_fn, critic_fn, "not found")
        # prev_weight = model.actor.layer.weight.clone()
        model.actor.load_state_dict(torch.load(actor_fn))
        model.critic.load_state_dict(torch.load(critic_fn))
        # curr_weight = model.actor.layer.weight
        exp = actor_fn.split("/")[-2]
        output_fn = "_".join(actor_fn.split("/")[-1].split(".")[0].split("_")[1:])
    else:
        output_fn = "vanilla"
    output_fn = os.path.join(
        "output", exp, output_fn + ".txt"
    )
    if os.path.exists(output_fn):
        return
        # os.remove(output_fn)
    generate_dir(os.path.dirname(output_fn))
    ppo_buffer = PPO(0.95, 0.99, env, model)
    trajectories = ppo_buffer.test(count, save_fn=output_fn)
    return


def check(fn, n=3):
    with open(fn, "r") as f:
        lines = f.readlines()
        smiles = [line.split()[0] for line in lines]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        ringinfos = [mol.GetRingInfo() for mol in mols]
        ringcount = [len(ringinfo.AtomRings()) for ringinfo in ringinfos]
        print(
            fn,
            "# smiles: ",
            len(smiles),
            "Total # molecules with ring: ",
            sum(ringcount),
        )
    return


def worker(files):
    print(files)
    actor_fn, critic_fn = files
    exp = actor_fn.split("/")[-2]
    # test(count=10000, exp=exp)
    exp = actor_fn.split("/")[-2]
    output_fn = "_".join(actor_fn.split("/")[-1].split(".")[0].split("_")[1:])
    if os.path.exists(output_fn):
        return
    test(fns=(actor_fn, critic_fn), count=10000)
    return


def mp(exps, ncpus):
    pool = Pool(ncpus)
    results = pool.map_async(worker, exps)
    results.wait()
    pool.close()
    pool.join()
    return


if __name__ == "__main__":
    seed_all(0)
    N = 3
    env = MoleculeEnv(N, N)
    actor = PolicyNetwork()
    critic = ValueNetwork()
    model = ActorCritic(actor, critic)
    exps = sorted(os.listdir("save"))
    # test(count=10000, exp=exps[1])
    actors = [f"./save/6-5/actor_{i*10}_0.pt" for i in range(27)]
    critics = [f"./save/6-5/critic_{i*10}_0.pt" for i in range(27)]
    files = list(zip(actors, critics))
    mp(files, 12)
    exit()
    # for exp in exps[:-1]:
    #     test(count=10000, exp=exp)
    # exit()
    i = sys.argv[1]
    # ncpu = int(sys.argv[2])
    ncpu = 16
    with torch.no_grad():
        actors = sorted(glob.glob(f"./save/{i}-*/actor*"))
        critics = sorted(glob.glob(f"./save/{i}-*/critic*"))
        actors = [actor for actor in actors if int(actor.split("/")[-1].split("_")[1]) < 31]
        critics = [critic for critic in critics if int(critic.split("/")[-1].split("_")[1]) < 31]
        files = list(zip(actors, critics))
        mp(files, ncpu)
        # mp(files[:16], ncpu)
        # mp(files[16:32], ncpu)
        # mp(files[32:48], ncpu)
        # mp(files[48:64], ncpu)
        # mp(files[64:80], ncpu)
        # mp(files[80:96], ncpu)
        # mp(files[96:112], ncpu)
        # mp(files[112:], ncpu)
