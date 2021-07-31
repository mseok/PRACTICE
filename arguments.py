import argparse
from pathlib import PosixPath

import data.loader as loader
import utils


def update_to_abs_path(args):
    update_dict = {}
    for key, value in vars(args).items():
        if type(value) == PosixPath:
            update_dict[key] = utils.get_abs_path(value)
        else:
            update_dict[key] = value
    new_args = argparse.Namespace(**update_dict)
    return new_args


def parser(command):
    arg_command = command[1:]
    parser = argparse.ArgumentParser()

    # PATH
    parser.add_argument("--data_dir", type=PosixPath)
    parser.add_argument("--key_dir", type=PosixPath)
    if "train" in command[0]:
        parser.add_argument("--save_dir", type=PosixPath)

    # Files
    parser.add_argument("--restart_fn", type=PosixPath)
    parser.add_argument("--log_fn", type=PosixPath)

    # Model
    parser.add_argument("--model", type=str, choices=["egnn", "schnet"])
    parser.add_argument("--n_dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--task", type=str, choices=loader.PROPERTIES)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--n_filters", type=int)
    parser.add_argument("--filter_spacing", type=float)

    # Main
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--ngpu", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    if "train" in command[0]:
        # Train
        parser.add_argument("--nr", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--num_epochs", type=int)
        parser.add_argument("--save_every", type=int)
        parser.add_argument("--dropout_rate", type=float)

    if "test" in command[0]:
        # Files
        parser.add_argument("--test_result_fn", type=PosixPath)

    FLAGS, UNUSED = parser.parse_known_args(arg_command)
    FLAGS = update_to_abs_path(FLAGS)
    return FLAGS, UNUSED
