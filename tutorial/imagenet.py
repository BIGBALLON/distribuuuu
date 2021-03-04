"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
"""

import os
import subprocess

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

BATCH_SIZE = 256
EPOCHS = 5
IMAGE_DIR = "./ImageNet/"


def setup_distributed(backend="nccl", port=None):
    """
    Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


if __name__ == "__main__":

    # 0. set up distributed device
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

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
    trainset = torchvision.datasets.ImageFolder(
        root=traindir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1 * dist.get_world_size(),
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    if rank == 0:
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

            if rank == 0 and ((idx + 1) % 20 == 0 or (idx + 1) == len(train_loader)):
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
example: 8GPUs (batch size: 2048)
>>> python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    imagenet.py

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
