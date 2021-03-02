"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.multiprocessing
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_launch.py] and find out the differences.
"""
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5


parser = argparse.ArgumentParser()
parser.add_argument(
    "--nodes", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--ngpus_per_node",
    default=2,
    type=int,
    help="number of GPUs per node for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:12306",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--node_rank", default=0, type=int, help="node rank for distributed training"
)


def main():
    args = parser.parse_args()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    # global_world_size = ngpus_per_node * nodes
    args.global_world_size = args.ngpus_per_node * args.nodes
    mp.spawn(train_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))


def train_worker(local_rank, ngpus_per_node, args):

    # 0. set up distributed device
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    # global_rank = node_rank * ngpus_per_node + local_rank
    args.global_rank = args.node_rank * ngpus_per_node + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.global_world_size,
        rank=args.global_rank,
    )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {args.global_rank} ==")

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
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01 * 2,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    if args.global_rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(0, EPOCHS):
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

            if args.global_rank == 0 and (
                (idx + 1) % 25 == 0 or (idx + 1) == len(train_loader)
            ):
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
    if args.global_rank == 0:
        print("\n            =======  Training Finished  ======= \n")


if __name__ == "__main__":
    main()


"""
usage:
>>> python mnmc_ddp_mp.py --help

exmaple:
>>> python mnmc_ddp_mp.py --nodes=1 --ngpus_per_node=2

[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 0, global rank: 0 ==
            =======  Training  ======= 

   == step: [ 25/98] [0/5] | loss: 2.020 | acc: 27.266%
   == step: [ 50/98] [0/5] | loss: 1.857 | acc: 32.266%
   == step: [ 75/98] [0/5] | loss: 1.761 | acc: 35.516%
   == step: [ 98/98] [0/5] | loss: 1.705 | acc: 37.668%
   == step: [ 25/98] [1/5] | loss: 1.438 | acc: 46.922%
   == step: [ 50/98] [1/5] | loss: 1.411 | acc: 48.305%
   == step: [ 75/98] [1/5] | loss: 1.385 | acc: 49.396%
   == step: [ 98/98] [1/5] | loss: 1.363 | acc: 50.292%
   == step: [ 25/98] [2/5] | loss: 1.259 | acc: 54.297%
   == step: [ 50/98] [2/5] | loss: 1.245 | acc: 54.773%
   == step: [ 75/98] [2/5] | loss: 1.230 | acc: 55.401%
   == step: [ 98/98] [2/5] | loss: 1.217 | acc: 55.944%
   == step: [ 25/98] [3/5] | loss: 1.159 | acc: 58.641%
   == step: [ 50/98] [3/5] | loss: 1.136 | acc: 59.320%
   == step: [ 75/98] [3/5] | loss: 1.121 | acc: 59.922%
   == step: [ 98/98] [3/5] | loss: 1.109 | acc: 60.400%
   == step: [ 25/98] [4/5] | loss: 1.011 | acc: 64.047%
   == step: [ 50/98] [4/5] | loss: 1.016 | acc: 63.398%
   == step: [ 75/98] [4/5] | loss: 1.009 | acc: 63.604%
   == step: [ 98/98] [4/5] | loss: 1.006 | acc: 64.084%

            =======  Training Finished  ======= 
"""
