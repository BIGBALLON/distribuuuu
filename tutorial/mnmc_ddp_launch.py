"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.distributed.launch
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_mp.py] and find out the differences.
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5


if __name__ == "__main__":

    # 0. set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # 1. define network
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.to(device)
    # DistributedDataParallel
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
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
        trainset,
        shuffle=True,
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
        net.parameters(),
        lr=0.01 * 2,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    if rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)

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

            if rank == 0 and ((idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)):
                print(
                    "   == step: [{:3}/{}] [{}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        ep,
                        EPOCHS,
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
    if rank == 0:
        print("\n            =======  Training Finished  ======= \n")

"""
usage:
>>> python -m torch.distributed.launch --help

exmaple: 1 node, 4 GPUs per node (4GPUs)
>>> python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    mnmc_ddp_launch.py

[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 2, global rank: 2 ==
            =======  Training  ======= 

   == step: [ 25/49] [0/5] | loss: 1.980 | acc: 27.953%
   == step: [ 49/49] [0/5] | loss: 1.806 | acc: 33.816%
   == step: [ 25/49] [1/5] | loss: 1.464 | acc: 47.391%
   == step: [ 49/49] [1/5] | loss: 1.420 | acc: 48.448%
   == step: [ 25/49] [2/5] | loss: 1.300 | acc: 52.469%
   == step: [ 49/49] [2/5] | loss: 1.274 | acc: 53.648%
   == step: [ 25/49] [3/5] | loss: 1.201 | acc: 56.547%
   == step: [ 49/49] [3/5] | loss: 1.185 | acc: 57.360%
   == step: [ 25/49] [4/5] | loss: 1.129 | acc: 59.531%
   == step: [ 49/49] [4/5] | loss: 1.117 | acc: 59.800%

            =======  Training Finished  =======

exmaple: 2 node, 4 GPUs per node (8GPUs)
>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py

>>> CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py

            =======  Training  ======= 

   == step: [ 25/25] [0/5] | loss: 1.932 | acc: 29.088%
   == step: [ 25/25] [1/5] | loss: 1.546 | acc: 43.088%
   == step: [ 25/25] [2/5] | loss: 1.424 | acc: 48.032%
   == step: [ 25/25] [3/5] | loss: 1.335 | acc: 51.440%
   == step: [ 25/25] [4/5] | loss: 1.243 | acc: 54.672%

            =======  Training Finished  =======

exmaple: 2 node, 8 GPUs per node (16GPUs)
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py

>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.198.189.10" \
    --master_port=22222 \
    mnmc_ddp_launch.py

[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 0, global rank: 0 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 1, global rank: 1 ==
            =======  Training  ======= 

   == step: [ 13/13] [0/5] | loss: 2.056 | acc: 23.776%
   == step: [ 13/13] [1/5] | loss: 1.688 | acc: 36.736%
   == step: [ 13/13] [2/5] | loss: 1.508 | acc: 44.544%
   == step: [ 13/13] [3/5] | loss: 1.462 | acc: 45.472%
   == step: [ 13/13] [4/5] | loss: 1.357 | acc: 49.344%

            =======  Training Finished  ======= 
"""
