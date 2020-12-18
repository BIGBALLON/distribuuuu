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

# each DDP instance will process BATCH_SIZE images.
# we test single Machine with 4 GPUs
# so BATCH_SIZE is set to 64 (256 / 4 = 64)

BATCH_SIZE = 64
EPOCHS = 10
LRSTEP = 5


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(torch.flatten(x, 1))
        return x


if __name__ == "__main__":

    # 0. set up distributed device
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        print("            =======  Distributed Settings  ======= \n")
        print(f" == local rank: {local_rank}")
        print(f" == device: {device}")
        print(" == backend: nccl")

    # 1. define netowrk
    net = AlexNet()
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
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=0.1)

    if local_rank == 0:
        print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        if local_rank == 0:
            print(f" === Epoch: [{ep}/{EPOCHS}] === ")

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

            if (
                local_rank == 0
                and (idx + 1) % 100 == 0
                or (idx + 1) == len(train_loader)
            ):
                print(
                    "   == step: [{:3}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )
        scheduler.step()
    if local_rank == 0:
        print("\n            =======  Training Finished  ======= \n")
