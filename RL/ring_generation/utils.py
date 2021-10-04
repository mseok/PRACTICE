import logging
import os
import random
import time

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolToImageFile
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def draw_mol(mol, fn):
    mol = mol_with_atom_index(mol)
    MolToImageFile(mol, fn)
    return None


def dic_to_device(dic, device):
    for dic_key, dic_value in dic.items():
        if isinstance(dic_value, torch.Tensor):
            dic_value = dic_value.to(device)
            dic[dic_key] = dic_value
    return dic


def to_device(batch, device):
    if isinstance(batch, dict):
        return {key: to_device(elem, device) for key, elem in batch.items()}
    elif isinstance(batch, list) or isinstance(batch, tuple):
        elem_type = type(batch)
        return elem_type((to_device(elem, device) for elem in batch))
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, Chem.Mol):
        return batch
    else:
        print(type(batch))
        raise TypeError("Batch should be dictionary or list")


def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        if device.type == "cpu":
            load = torch.load(load_save_file, map_location="cpu")
        else:
            load = torch.load(load_save_file)
        model.load_state_dict(load, strict=True)
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                nn.init.xavier_normal_(param)
    model.to(device)
    return model


# Path
def get_abs_path(path):
    return os.path.realpath(os.path.expanduser(path))


def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return


def check_current_enable_device():
    from subprocess import Popen, PIPE

    for i in range(8):
        pipe = Popen(["nvidia-smi", "-i", str(i)], stdout=PIPE)
        pipe = Popen(["grep", "No running"], stdin=pipe.stdout, stdout=PIPE)
        pipe = Popen(
            ["wc", "-l"],
            stdin=pipe.stdout,
            stdout=PIPE,
            encoding="utf-8",
        )
        output = int(pipe.communicate()[0].split("\n")[0])
        if output:
            break
    return i


# Debug
def debug_anomaly(use_anomaly: bool):
    def anomaly(func):
        def wrapper(*args, **kwargs):
            with torch.autograd.set_detect_anomaly(use_anomaly):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return anomaly


def check_time(*args):
    def get_time(func):
        def wrapper(*args, **kwargs):
            st = time.time()
            result = func(*args, **kwargs)
            print(time.time() - st)

        return wrapper

    return get_time


def compare_params(param1, param2):
    for p1, p2 in zip(param1, param2):
        print(torch.all(p1 == p2))
    return


def stat_cuda(msg):
    print("--", msg)
    print(
        "allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM"
        % (
            torch.cuda.memory_allocated() / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            torch.cuda.memory_cached() / 1024 / 1024,
            torch.cuda.max_memory_cached() / 1024 / 1024,
        )
    )


# logger
def init_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != "":
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10
            )
        else:
            file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def log_arguments(logger, args):
    if logger is None:
        return
    logger.info(f"Current Working Directory: {os.getcwd()}")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")


def log_msg(logger, msg):
    if logger is None:
        return
    logger.info(msg)
