"""
Single Machine Multi-GPU Training (with DistributedDataParallel)
Try to compare with [smsc.py, smmc_dp.py] and find out the differences.
"""

import argparse

import torch
import torch.distributed as dist  # for distributed training
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5
BACKEND = "nccl"

if __name__ == "__main__":

    # 0. set up distributed device
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=BACKEND)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print("            =======  Distributed Settings  ======= \n")
        print(f" == local rank: {local_rank}")
        print(f" == device: {device}")
        print(f" == backend: {BACKEND}")

    # 1. define netowrk
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    # SyncBN
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    # DistributedSampler
    # we test single Machine with 2 GPUs
    # so the [batch size] for each process is 256 / 2 = 128
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.01 * 2, momentum=0.9, weight_decay=0.0001, nesterov=True,
    )

    if local_rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(0, EPOCHS):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)
        if local_rank == 0:
            print(f" === Epoch: [{ep + 1}/{EPOCHS}] === ")

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if local_rank == 0 and (
                (idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)
            ):
                print(
                    "   == step: [{:3}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
    if local_rank == 0:
        print("\n            =======  Training Finished  ======= \n")
