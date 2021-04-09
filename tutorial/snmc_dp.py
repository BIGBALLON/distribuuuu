"""
(SNMC) Single Node Multi-GPU Cards Training (with DataParallel)
Try to compare with smsc.py and find out the differences.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 5

if __name__ == "__main__":

    # 1. define network
    device = "cuda"
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.to(device=device)
    # Use single-machine multi-GPU DataParallel,
    # you would like to speed up training with the minimum code change.
    net = nn.DataParallel(net)

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
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0

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

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
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

    print("\n            =======  Training Finished  ======= \n")

"""
usage: 2GPUs for training
>>> CUDA_VISIBLE_DEVICES=0,1 python snmc_dp.py

Files already downloaded and verified
            =======  Training  ======= 

   == step: [ 50/196] [1/5] | loss: 1.992 | acc: 26.633%
   == step: [100/196] [1/5] | loss: 1.834 | acc: 32.797%
   == step: [150/196] [1/5] | loss: 1.742 | acc: 36.201%
   == step: [196/196] [1/5] | loss: 1.680 | acc: 38.578%
   == step: [ 50/196] [2/5] | loss: 1.398 | acc: 49.062%
   == step: [100/196] [2/5] | loss: 1.380 | acc: 49.953%
   == step: [150/196] [2/5] | loss: 1.355 | acc: 50.810%
   == step: [196/196] [2/5] | loss: 1.338 | acc: 51.428%
   == step: [ 50/196] [3/5] | loss: 1.242 | acc: 55.727%
   == step: [100/196] [3/5] | loss: 1.219 | acc: 56.801%
   == step: [150/196] [3/5] | loss: 1.200 | acc: 57.195%
   == step: [196/196] [3/5] | loss: 1.193 | acc: 57.328%
   == step: [ 50/196] [4/5] | loss: 1.105 | acc: 61.102%
   == step: [100/196] [4/5] | loss: 1.098 | acc: 61.082%
   == step: [150/196] [4/5] | loss: 1.087 | acc: 61.354%
   == step: [196/196] [4/5] | loss: 1.086 | acc: 61.426%
   == step: [ 50/196] [5/5] | loss: 1.002 | acc: 64.039%
   == step: [100/196] [5/5] | loss: 1.006 | acc: 63.977%
   == step: [150/196] [5/5] | loss: 1.009 | acc: 63.935%
   == step: [196/196] [5/5] | loss: 1.005 | acc: 64.024%

            =======  Training Finished  ======= 
"""
