"""(SNSC) Single Node Single GPU Card Training"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 5

if __name__ == "__main__":

    # 1. define network
    device = "cuda"
    net = torchvision.models.resnet18(pretrained=True, num_classes=10)
    net = net.to(device=device)

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
usage:
>>> python snsc.py

Files already downloaded and verified
            =======  Training  ======= 

   == step: [ 50/196] [1/5] | loss: 1.959 | acc: 28.633%
   == step: [100/196] [1/5] | loss: 1.806 | acc: 33.996%
   == step: [150/196] [1/5] | loss: 1.718 | acc: 36.987%
   == step: [196/196] [1/5] | loss: 1.658 | acc: 39.198%
   == step: [ 50/196] [2/5] | loss: 1.393 | acc: 49.578%
   == step: [100/196] [2/5] | loss: 1.359 | acc: 50.473%
   == step: [150/196] [2/5] | loss: 1.336 | acc: 51.372%
   == step: [196/196] [2/5] | loss: 1.317 | acc: 52.200%
   == step: [ 50/196] [3/5] | loss: 1.205 | acc: 56.102%
   == step: [100/196] [3/5] | loss: 1.185 | acc: 57.254%
   == step: [150/196] [3/5] | loss: 1.175 | acc: 57.755%
   == step: [196/196] [3/5] | loss: 1.165 | acc: 58.072%
   == step: [ 50/196] [4/5] | loss: 1.067 | acc: 60.914%
   == step: [100/196] [4/5] | loss: 1.061 | acc: 61.406%
   == step: [150/196] [4/5] | loss: 1.058 | acc: 61.643%
   == step: [196/196] [4/5] | loss: 1.054 | acc: 62.022%
   == step: [ 50/196] [5/5] | loss: 0.988 | acc: 64.852%
   == step: [100/196] [5/5] | loss: 0.983 | acc: 64.801%
   == step: [150/196] [5/5] | loss: 0.980 | acc: 65.052%
   == step: [196/196] [5/5] | loss: 0.977 | acc: 65.076%

            =======  Training Finished  ======= 
"""
