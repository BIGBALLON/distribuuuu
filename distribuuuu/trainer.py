import os
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from distribuuuu import models, utils
from distribuuuu.config import cfg


def train_epoch(train_loader, net, criterion, optimizer, cur_epoch, rank):

    batch_time, data_time, losses, top1, topk = utils.get_meters()
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix=" = TRAIN:     [{}]".format(cur_epoch),
    )

    lr = utils.get_epoch_lr(cur_epoch)
    utils.set_lr(optimizer, lr)

    # set sampler
    train_loader.sampler.set_epoch(cur_epoch)
    # switch to train mode
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
        acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, 5))
        loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])

        losses.update(loss.item(), batch_size)
        top1.update(acc_1[0], batch_size)
        topk.update(acc_k[0], batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and (
            idx % cfg.TRAIN.PRINT_FEQ == 0 or (idx + 1) == len(train_loader)
        ):
            progress.display(idx)


def validate(val_loader, net, criterion, cur_epoch, rank):

    batch_time, data_time, losses, top1, topk = utils.get_meters()
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix=" = VAL:     [{}]".format(cur_epoch),
    )

    # switch to evaluate mode
    net.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, 5))
            loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])

            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc_1[0], batch_size)
            topk.update(acc_k[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0 and (
                idx % cfg.TEST.PRINT_FEQ == 0 or (idx + 1) == len(val_loader)
            ):
                progress.display(idx)

    return top1.avg, topk.avg


def train_model():

    # 0x00. set up distributed device
    utils.setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    utils.show_log(f" = INFO:     LOCAL_RANK: {local_rank}, RANK: {rank}", 0)

    # 0x01. define network
    net = models.build_model(arch=cfg.MODEL.ARCH, pretrained=False, num_classes=1000)
    # SyncBN
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    # 0x02. define dataloader
    train_loader, val_loader = utils.construct_loader()

    # 0x03. define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = utils.construct_optimizer(net)

    # optionally resume from a checkpoint
    best_acc1 = start_epoch = 0
    if cfg.TRAIN.CHECKPOINT:
        if os.path.isfile(cfg.TRAIN.CHECKPOINT):
            # Map model to be loaded to specified single gpu.
            map_location = f"cuda:{local_rank}"
            checkpoint = torch.load(cfg.TRAIN.CHECKPOINT, map_location=map_location)
            start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            utils.show_log(
                f" = INFO:     LOADED '{cfg.TRAIN.CHECKPOINT}' (epoch {start_epoch})",
                rank,
            )
        else:
            utils.show_log(
                f" = WARNING:     NO CHECKPOINT FOUND AT '{cfg.TRAIN.CHECKPOINT}'", rank
            )

    utils.show_log("\n            =======  TRAINING  ======= \n", rank)

    # 0x04. start to train
    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        train_epoch(train_loader, net, criterion, optimizer, epoch, rank)
        acc1, acc5 = validate(val_loader, net, criterion, epoch, rank)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if rank == 0:
            ckpt_name = f"ckpt_{epoch}.pth.tar"
            utils.save_checkpoint(
                {
                    "epoch": epoch,
                    "arch": cfg.MODEL.ARCH,
                    "state_dict": net.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename=ckpt_name,
            )
            utils.show_log(
                f" = INFO:     Acc@1 {acc1:.3f} | Acc@5 {acc5:.3f} | BEST Acc@1 {best_acc1:.3f} | SAVED {ckpt_name}",
                rank,
            )
