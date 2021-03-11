"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
Minimal ImageNet training code powered by DDP
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
EPOCHS = 1
IMAGE_DIR = "./data/ILSVRC/"


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
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
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

            if rank == 0 and ((idx + 1) % 40 == 0 or (idx + 1) == len(train_loader)):
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

    # 5. save model (only in rank0)
    checkpoint_file = "./ckpt.pth.tar"
    if rank == 0:
        checkpoint = {
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"(rank: {rank})   == Saved: {checkpoint_file}")

    # 6. load model (all ranks)
    # use a barrier() to make sure that other ranks loads the model after rank0 saves it.
    # see https://github.com/pytorch/examples/blob/master/distributed/ddp/main.py
    dist.barrier()
    map_location = f"cuda:{local_rank}"
    # map model to be loaded to specified single gpu.
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"(rank: {rank})   == Loaded: {checkpoint_file}")

"""
distributed.launch example: 
    8GPUs (batch size: 2048)
    128k / (256*8) -> 626 itertaion

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
   == step: [ 40/626] [0/1] | loss: 6.821 | acc:  0.498%
   == step: [ 80/626] [0/1] | loss: 6.616 | acc:  0.869%
   == step: [120/626] [0/1] | loss: 6.448 | acc:  1.351%
   == step: [160/626] [0/1] | loss: 6.294 | acc:  1.868%
   == step: [200/626] [0/1] | loss: 6.167 | acc:  2.443%
   == step: [240/626] [0/1] | loss: 6.051 | acc:  3.003%
   == step: [280/626] [0/1] | loss: 5.952 | acc:  3.457%
   == step: [320/626] [0/1] | loss: 5.860 | acc:  3.983%
   == step: [360/626] [0/1] | loss: 5.778 | acc:  4.492%
   == step: [400/626] [0/1] | loss: 5.700 | acc:  4.960%
   == step: [440/626] [0/1] | loss: 5.627 | acc:  5.488%
   == step: [480/626] [0/1] | loss: 5.559 | acc:  6.013%
   == step: [520/626] [0/1] | loss: 5.495 | acc:  6.520%
   == step: [560/626] [0/1] | loss: 5.429 | acc:  7.117%
   == step: [600/626] [0/1] | loss: 5.371 | acc:  7.580%
   == step: [626/626] [0/1] | loss: 5.332 | acc:  7.907%

(rank: 0)   == Saved: ./ckpt.pth.tar
(rank: 0)   == Loaded: ./ckpt.pth.tar
(rank: 1)   == Loaded: ./ckpt.pth.tar
(rank: 6)   == Loaded: ./ckpt.pth.tar
(rank: 7)   == Loaded: ./ckpt.pth.tar
(rank: 4)   == Loaded: ./ckpt.pth.tar
(rank: 5)   == Loaded: ./ckpt.pth.tar
(rank: 3)   == Loaded: ./ckpt.pth.tar
(rank: 2)   == Loaded: ./ckpt.pth.tar


slurm example: 
    32GPUs (batch size: 8192)
    128k / (256*32) -> 157 itertaion
>>> srun --partition=openai -n32 --gres=gpu:8 --ntasks-per-node=8 --job-name=slrum_test \
    python -u imagenet.py

[init] == local rank: 7, global rank: 7 ==
[init] == local rank: 1, global rank: 1 ==
[init] == local rank: 4, global rank: 4 ==
[init] == local rank: 2, global rank: 2 ==
[init] == local rank: 6, global rank: 6 ==
[init] == local rank: 3, global rank: 3 ==
[init] == local rank: 5, global rank: 5 ==
[init] == local rank: 4, global rank: 12 ==
[init] == local rank: 1, global rank: 25 ==
[init] == local rank: 5, global rank: 13 ==
[init] == local rank: 6, global rank: 14 ==
[init] == local rank: 0, global rank: 8 ==
[init] == local rank: 1, global rank: 9 ==
[init] == local rank: 2, global rank: 10 ==
[init] == local rank: 3, global rank: 11 ==
[init] == local rank: 7, global rank: 15 ==
[init] == local rank: 5, global rank: 29 ==
[init] == local rank: 2, global rank: 26 ==
[init] == local rank: 3, global rank: 27 ==
[init] == local rank: 0, global rank: 24 ==
[init] == local rank: 7, global rank: 31 ==
[init] == local rank: 6, global rank: 30 ==
[init] == local rank: 4, global rank: 28 ==
[init] == local rank: 0, global rank: 16 ==
[init] == local rank: 5, global rank: 21 ==
[init] == local rank: 7, global rank: 23 ==
[init] == local rank: 1, global rank: 17 ==
[init] == local rank: 6, global rank: 22 ==
[init] == local rank: 3, global rank: 19 ==
[init] == local rank: 2, global rank: 18 ==
[init] == local rank: 4, global rank: 20 ==
[init] == local rank: 0, global rank: 0 ==
            =======  Training  ======= 

   == step: [ 40/157] [0/1] | loss: 6.781 | acc:  0.703%
   == step: [ 80/157] [0/1] | loss: 6.536 | acc:  1.260%
   == step: [120/157] [0/1] | loss: 6.353 | acc:  1.875%
   == step: [157/157] [0/1] | loss: 6.207 | acc:  2.465%

(rank: 0)   == Saved: ./ckpt.pth.tar
(rank: 0)   == Loaded: ./ckpt.pth.tar
(rank: 6)   == Loaded: ./ckpt.pth.tar
(rank: 2)   == Loaded: ./ckpt.pth.tar
(rank: 1)   == Loaded: ./ckpt.pth.tar
(rank: 5)   == Loaded: ./ckpt.pth.tar
(rank: 7)   == Loaded: ./ckpt.pth.tar
(rank: 3)   == Loaded: ./ckpt.pth.tar
(rank: 4)   == Loaded: ./ckpt.pth.tar
(rank: 11)   == Loaded: ./ckpt.pth.tar
(rank: 9)   == Loaded: ./ckpt.pth.tar
(rank: 8)   == Loaded: ./ckpt.pth.tar
(rank: 14)   == Loaded: ./ckpt.pth.tar
(rank: 12)   == Loaded: ./ckpt.pth.tar
(rank: 15)   == Loaded: ./ckpt.pth.tar
(rank: 13)   == Loaded: ./ckpt.pth.tar
(rank: 10)   == Loaded: ./ckpt.pth.tar
(rank: 21)   == Loaded: ./ckpt.pth.tar
(rank: 23)   == Loaded: ./ckpt.pth.tar
(rank: 20)   == Loaded: ./ckpt.pth.tar
(rank: 17)   == Loaded: ./ckpt.pth.tar
(rank: 19)   == Loaded: ./ckpt.pth.tar
(rank: 16)   == Loaded: ./ckpt.pth.tar
(rank: 18)   == Loaded: ./ckpt.pth.tar
(rank: 22)   == Loaded: ./ckpt.pth.tar
(rank: 29)   == Loaded: ./ckpt.pth.tar
(rank: 31)   == Loaded: ./ckpt.pth.tar
(rank: 24)   == Loaded: ./ckpt.pth.tar
(rank: 27)   == Loaded: ./ckpt.pth.tar
(rank: 30)   == Loaded: ./ckpt.pth.tar
(rank: 28)   == Loaded: ./ckpt.pth.tar
(rank: 26)   == Loaded: ./ckpt.pth.tar
(rank: 25)   == Loaded: ./ckpt.pth.tar

"""
