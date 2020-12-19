"""Single Machine Single GPU Training"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 5

if __name__ == "__main__":

    # 1. define netowrk
    device = "cuda"
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
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
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True,
    )

    print("            =======  Training  ======= \n")

    # 4. start to train
    net.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
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

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print(
                    "   == step: [{:3}/{}] | loss: {:.3f} | acc: {:6.3f}%".format(
                        idx + 1,
                        len(train_loader),
                        train_loss / (idx + 1),
                        100.0 * correct / total,
                    )
                )

    print("\n            =======  Training Finished  ======= \n")
