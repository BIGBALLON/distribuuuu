"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
"""

import argparse
import os
import subprocess
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5
IMAGE_DIR = "./ImageNet/"
CKPT = "model_best.pth.tar"

from utils import (
    ProgressMeter,
    accuracy,
    get_meters,
    save_checkpoint,
    scaled_all_reduce,
    setup_distributed,
)


def train_epoch(train_loader, net, criterion, optimizer, ep, rank):

    batch_time, data_time, losses, top1, topk = get_meters()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix="Epoch: [{}]".format(ep),
    )

    # set sampler
    train_loader.sampler.set_epoch(ep)
    # switch to train mode
    net.train()

    end = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        acc_1, acc_k = accuracy(outputs, targets, topk=(1, 5))
        loss, acc_1, acc_k = scaled_all_reduce([loss, acc_1, acc_k])

        losses.update(loss.item(), batch_size)
        top1.update(acc_1[0], batch_size)
        topk.update(acc_k[0], batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and (idx % 10 == 0 or (idx + 1) == len(train_loader)):
            progress.display(idx)


def validate(val_loader, net, criterion, rank):

    batch_time, data_time, losses, top1, topk = get_meters()
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix="Test: [{}]".format(ep),
    )

    # switch to evaluate mode
    net.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            acc_1, acc_k = accuracy(outputs, targets, topk=(1, 5))
            loss, acc_1, acc_k = scaled_all_reduce([loss, acc_1, acc_k])

            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc_1[0], batch_size)
            topk.update(acc_k[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0 and (idx % 10 == 0 or (idx + 1) == len(val_loader)):
                progress.display(idx)

    if rank == 0:
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {topk.avg:.3f}".format(top1=top1, topk=topk)
        )
    return top1.avg


if __name__ == "__main__":

    # 0. set up distributed device
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    best_acc1 = 0

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=1000)
    # SyncBN
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    # 2. define dataloader
    traindir = os.path.join(IMAGE_DIR, "train")
    valdir = os.path.join(IMAGE_DIR, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    trainset = torchvision.datasets.ImageFolder(
        root=traindir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    # DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=valdir,
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1 * dist.get_world_size(),
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    start_epoch = 1
    if CKPT:
        if os.path.isfile(CKPT):
            print("=> loading checkpoint '{}'".format(CKPT))
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(local_rank)
            checkpoint = torch.load(CKPT, map_location=loc)
            start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(CKPT, checkpoint["epoch"])
            )
        else:
            print("=> no checkpoint found at '{}'".format(CKPT))

    if rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    for ep in range(start_epoch, EPOCHS + 1):
        train_epoch(train_loader, net, criterion, optimizer, ep, rank)
        acc1 = validate(val_loader, net, criterion, rank)
        scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if rank == 0:
            save_checkpoint(
                {
                    "epoch": ep,
                    "arch": "resnet18",
                    "state_dict": net.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename=f"ckpt_{ep}.pth.tar",
            )

"""
example: 8GPUs (batch size: 2048)
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    trainer.py

[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 1, global rank: 1 ==
            =======  Training  ======= 

   == step: [ 20/626] [0/5] | loss: 6.943 | acc:  0.293%
   == step: [ 40/626] [0/5] | loss: 6.794 | acc:  0.420%
   == step: [ 60/626] [0/5] | loss: 6.671 | acc:  0.762%
   == step: [ 80/626] [0/5] | loss: 6.572 | acc:  0.894%
   == step: [100/626] [0/5] | loss: 6.491 | acc:  1.145%
   == step: [120/626] [0/5] | loss: 6.411 | acc:  1.377%
   == step: [140/626] [0/5] | loss: 6.332 | acc:  1.613%
   == step: [160/626] [0/5] | loss: 6.259 | acc:  1.953%
   == step: [180/626] [0/5] | loss: 6.194 | acc:  2.263%
   == step: [200/626] [0/5] | loss: 6.128 | acc:  2.551%
   == step: [220/626] [0/5] | loss: 6.064 | acc:  2.862%
   == step: [240/626] [0/5] | loss: 6.002 | acc:  3.175%
   == step: [260/626] [0/5] | loss: 5.947 | acc:  3.483%
   == step: [280/626] [0/5] | loss: 5.894 | acc:  3.792%
   == step: [300/626] [0/5] | loss: 5.841 | acc:  4.133%
   == step: [320/626] [0/5] | loss: 5.786 | acc:  4.476%
   == step: [340/626] [0/5] | loss: 5.737 | acc:  4.816%
   == step: [360/626] [0/5] | loss: 5.688 | acc:  5.165%
   == step: [380/626] [0/5] | loss: 5.640 | acc:  5.516%
   == step: [400/626] [0/5] | loss: 5.593 | acc:  5.854%
   == step: [420/626] [0/5] | loss: 5.548 | acc:  6.192%
   == step: [440/626] [0/5] | loss: 5.502 | acc:  6.589%


>>> srun --partition=openai -n8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=slrum_test \
    python -u imagenet.py
"""
