import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import arguments
import data.loader as loader
from models.EGNN import EGNN
from models.SchNet import SchNet
import utils


class BestModel():
    def __init__(self, save_dir):
        self.loss = 1e10
        self.epoch = 0
        self.save_dir = save_dir

    def _update_model(self, val_loss, epoch, model):
        if self.loss > val_loss:
            self.loss = val_loss
            self.epoch = epoch
            name = os.path.join(self.save_dir, "best.pt")
            torch.save(model.state_dict(), name)
        return


@utils.debug_anomaly(False)
def train(model, data, device, loss_fn, optimizer):
    model.train()
    losses = []
    i_batch = 0
    while True:
        model.zero_grad()
        sample = next(data, None)
        if sample is None:
            break
        sample = utils.dic_to_device(sample, device)
        node_feat, edge_feat, adj, valid, pos, true, keys = sample.values()
        pred = model(sample)
        loss = loss_fn(pred, true)
        losses.append(loss.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i_batch += 1
    mean_loss = np.mean(np.array(losses))
    return mean_loss


def val(model, data, device, loss_fn):
    model.eval()
    losses = []
    i_batch = 0
    while True:
        model.zero_grad()
        sample = next(data, None)
        if sample is None:
            break
        sample = utils.dic_to_device(sample, device)
        node_feat, edge_feat, adj, valid, pos, true, keys = sample.values()
        pred = model(sample)
        loss = loss_fn(pred, true)
        losses.append(loss.data.cpu().numpy())
        i_batch += 1
    mean_loss = np.mean(np.array(losses))
    return mean_loss


def worker(gpu_idx, ngpus_per_node, FLAGS):
    # Default Settings
    torch.cuda.set_device(gpu_idx)
    rank = FLAGS.nr * FLAGS.ngpu + gpu_idx
    enable_log = (FLAGS.is_distributed and rank == 0) or not FLAGS.is_distributed
    if ngpus_per_node > 1:
        os.environ["MATER_ADDR"] = "127.0.0.1"
        os.environ["MATER_PORT"] = "2021"
        dist.init_process_group("nccl", rank=rank, world_size=FLAGS.world_size)

    # Path
    utils.generate_dir(FLAGS.save_dir)
    utils.generate_dir(os.path.dirname(FLAGS.log_fn))

    # Logger
    logger = utils.init_logger(FLAGS.log_fn) if enable_log else None
    utils.log_arguments(logger, FLAGS)

    # Update
    best_model = BestModel(FLAGS.save_dir)

    # Keys
    train_keys = utils.read_keys(FLAGS.key_dir, type="train")
    val_keys = utils.read_keys(FLAGS.key_dir, type="val")

    # Datalaoders
    train_dataset, train_loader, train_sampler \
        = loader.get_data(train_keys, ngpus_per_node, True, FLAGS)
    val_dataset, val_loader, val_sampler \
        = loader.get_data(val_keys, ngpus_per_node, False, FLAGS)
    utils.log_msg(logger, f"Train Size: {len(train_dataset)}")
    utils.log_msg(logger, f"Val Size: {len(val_dataset)}")

    # Model
    if FLAGS.model == "egnn":
        model = EGNN(loader.N_NODE_FEATURES, FLAGS.n_dim, FLAGS.n_layers, False)
    elif FLAGS.model == "schnet":
        model = SchNet(loader.N_NODE_FEATURES, FLAGS.n_dim, FLAGS.n_layers,
                       FLAGS.gamma, FLAGS.n_filters, FLAGS.filter_spacing)
    model = utils.initialize_model(model, gpu_idx, FLAGS.restart_fn)
    # model.cuda(gpu_idx)
    if FLAGS.is_distributed:
        model = DDP(model, device_ids=[gpu_idx], find_unused_parameters=True)
        cudnn.benchmark = True
    n_parameters = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    utils.log_msg(logger, f"Number of parameters: {n_parameters}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    # Loss function
    loss_fn = nn.MSELoss()

    # Train
    if FLAGS.restart_fn:
        restart_epoch = int(FLAGS.restart_fn.split("_")[-1].split(".")[0])
    else:
        restart_epoch = 0

    if restart_epoch == 0:
        msg = ["epoch", "tl", "vl"]
        msg += ["time"]
        msg = [m.ljust(7) for m in msg]
        msg = "\t".join(msg)
        utils.log_msg(logger, msg)
    for epoch in range(restart_epoch, restart_epoch + FLAGS.num_epochs):
        if FLAGS.is_distributed:
            train_sampler.set_epoch(epoch)
        train_data = iter(train_loader)
        val_data = iter(val_loader)

        st = time.time()
        train_loss = train(model, train_data, gpu_idx, loss_fn, optimizer)
        val_loss = val(model, val_data, gpu_idx, loss_fn)
        et = time.time()
        msg = f"{epoch}\t{train_loss:.3f}\t{val_loss:.3f}\t"
        msg += f"{et - st:.3f}"
        msg = msg.split("\t")
        msg = [m.ljust(7) for m in msg]
        msg = "\t".join(msg)
        utils.log_msg(logger, msg)

        # Save Model
        name = os.path.join(FLAGS.save_dir, f"save_{epoch}.pt")
        save_every = 1 if not FLAGS.save_every else FLAGS.save_every
        if epoch % save_every == 0 and enable_log:
            torch.save(model.state_dict(), name)
        best_model._update_model(val_loss, epoch, model)
    return


def main():
    FLAGS, _ = arguments.parser(sys.argv)
    utils.seed_all(FLAGS.seed)
    cudnn.deterministic = True
    if FLAGS.ngpu > 1:
        FLAGS.world_size = FLAGS.ngpu
        FLAGS.is_disbributed = True
        mp.spawn(worker, nprocs=FLAGS.ngpu, args=(FLAGS.ngpu, FLAGS,))
    else:
        idx = utils.check_current_enable_device()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        FLAGS.is_distributed = False
        worker(0, FLAGS.ngpu, FLAGS)
    return


if __name__ == "__main__":
    main()
