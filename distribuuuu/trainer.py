import os
import time

import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from distribuuuu import models, utils
from distribuuuu.config import cfg


def train_epoch(train_loader, net, criterion, optimizer, cur_epoch, rank):

    batch_time, data_time, losses, top1, topk = utils.construct_meters()
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix=f"TRAIN:  [{cur_epoch+1}]",
    )

    # Set learning rate
    lr = utils.get_epoch_lr(cur_epoch)
    utils.set_lr(optimizer, lr)
    if rank == 0:
        logger.debug(f"CURRENT EPOCH: {cur_epoch+1:3d},   LR: {lr:.4f}")

    # Set sampler
    train_loader.sampler.set_epoch(cur_epoch)

    # Switch to train mode
    net.train()
    end = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, cfg.TRAIN.TOPK))
        loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])

        losses.update(loss.item(), batch_size)
        top1.update(acc_1[0], batch_size)
        topk.update(acc_k[0], batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and (
            (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0 or (idx + 1) == len(train_loader)
        ):
            progress.display(idx + 1)


def validate(val_loader, net, criterion, cur_epoch, rank):

    batch_time, data_time, losses, top1, topk = utils.construct_meters()
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix=f"VAL:  [{cur_epoch+1}]",
    )

    # Switch to evaluate mode
    net.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, cfg.TRAIN.TOPK))
            loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])

            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc_1[0], batch_size)
            topk.update(acc_k[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0 and (
                (idx + 1) % cfg.TEST.PRINT_FREQ == 0 or (idx + 1) == len(val_loader)
            ):
                progress.display(idx + 1)

    return top1.avg, topk.avg


def train_model():

    # 0x00. Set up distributed device
    utils.setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    logger.debug(f"LOCAL_RANK: {local_rank}, RANK: {rank}")
    utils.construct_logger()

    # 0x01. Define network
    net = models.build_model(arch=cfg.MODEL.ARCH)
    # SyncBN
    # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    # DistributedDataParallel wrapper
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 0x02. Define dataloader
    train_loader, val_loader = utils.construct_loader()

    # 0x03. Define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = utils.construct_optimizer(net)

    # Resume from a specific checkpoint or the last checkpoint
    best_acc1 = start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and utils.has_checkpoint():
        file = utils.get_last_checkpoint()
        start_epoch, best_acc1 = utils.load_checkpoint(file, net, optimizer)
    elif cfg.TRAIN.WEIGHTS:
        load_opt = optimizer if cfg.TRAIN.LOAD_OPT else None
        utils.load_checkpoint(cfg.TRAIN.WEIGHTS, net, load_opt)

    if rank == 0:
        # from torch.utils.collect_env import get_pretty_env_info
        # logger.debug(get_pretty_env_info())
        # logger.debug(net)
        logger.info("\n\n\n            =======  TRAINING  ======= \n\n")

    # 0x04. Start to train
    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train one epoch
        train_epoch(train_loader, net, criterion, optimizer, epoch, rank)
        # Validate
        acc1, acck = validate(val_loader, net, criterion, epoch, rank)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # Save model
        checkpoint_file = utils.save_checkpoint(
            net, optimizer, epoch, best_acc1, is_best
        )
        if rank == 0:
            logger.info(
                f"ACCURACY:  TOP1 {acc1:.3f}(BEST {best_acc1:.3f}) | TOP{cfg.TRAIN.TOPK} {acck:.3f} | SAVED {checkpoint_file}"
            )
