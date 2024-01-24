"""
(SNMC) Single Node Multi-GPU Cards Training (with DDP)
Try to compare with smsc.py and find out the differences.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 256
EPOCHS = 5

if __name__ == "__main__":

    # 0. define ddp config 
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank() # single node multi-gpu cards by ddp
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 1. define network
    
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net = net.to(device=device)
    # Use single-machine multi-GPU DataParallel,
    # you would like to speed up training with the minimum code change.
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        net = torch.nn.parallel.DistributedDataParallel(net,
            device_ids=[local_rank], output_device=local_rank)

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
usage: 2GPUs for training by DDP
>>> CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 test.py

Let's use 2 GPUs!
Let's use 2 GPUs!
Files already downloaded and verified
Files already downloaded and verified
            =======  Training  ======= 

            =======  Training  ======= 

   == step: [ 50/196] [1/5] | loss: 1.904 | acc: 30.266%
   == step: [ 50/196] [1/5] | loss: 1.893 | acc: 30.242%
   == step: [100/196] [1/5] | loss: 1.732 | acc: 36.383%
   == step: [100/196] [1/5] | loss: 1.729 | acc: 36.500%
   == step: [150/196] [1/5] | loss: 1.636 | acc: 39.878%
   == step: [150/196] [1/5] | loss: 1.635 | acc: 39.898%
   == step: [196/196] [1/5] | loss: 1.573 | acc: 42.348%
   == step: [196/196] [1/5] | loss: 1.568 | acc: 42.442%
   == step: [ 50/196] [2/5] | loss: 1.293 | acc: 53.117%
   == step: [ 50/196] [2/5] | loss: 1.318 | acc: 52.430%
   == step: [100/196] [2/5] | loss: 1.264 | acc: 54.000%
   == step: [100/196] [2/5] | loss: 1.269 | acc: 54.133%
   == step: [150/196] [2/5] | loss: 1.238 | acc: 55.138%
   == step: [150/196] [2/5] | loss: 1.240 | acc: 55.086%
   == step: [196/196] [2/5] | loss: 1.215 | acc: 56.096%
   == step: [196/196] [2/5] | loss: 1.216 | acc: 56.020%
   == step: [ 50/196] [3/5] | loss: 1.088 | acc: 61.094%
   == step: [ 50/196] [3/5] | loss: 1.088 | acc: 60.406%
   == step: [100/196] [3/5] | loss: 1.067 | acc: 61.562%
   == step: [100/196] [3/5] | loss: 1.072 | acc: 61.660%
   == step: [150/196] [3/5] | loss: 1.050 | acc: 62.344%
   == step: [150/196] [3/5] | loss: 1.060 | acc: 62.177%
   == step: [196/196] [3/5] | loss: 1.040 | acc: 62.766%
   == step: [196/196] [3/5] | loss: 1.046 | acc: 62.586%
   == step: [ 50/196] [4/5] | loss: 0.948 | acc: 66.289%
   == step: [ 50/196] [4/5] | loss: 0.966 | acc: 65.469%
   == step: [100/196] [4/5] | loss: 0.939 | acc: 66.551%
   == step: [100/196] [4/5] | loss: 0.961 | acc: 65.645%
   == step: [150/196] [4/5] | loss: 0.932 | acc: 66.781%
   == step: [150/196] [4/5] | loss: 0.954 | acc: 65.901%
   == step: [196/196] [4/5] | loss: 0.929 | acc: 66.884%
   == step: [196/196] [4/5] | loss: 0.942 | acc: 66.406%
   == step: [ 50/196] [5/5] | loss: 0.902 | acc: 68.008%
   == step: [ 50/196] [5/5] | loss: 0.869 | acc: 69.164%
   == step: [100/196] [5/5] | loss: 0.884 | acc: 68.617%
   == step: [100/196] [5/5] | loss: 0.870 | acc: 69.145%
   == step: [150/196] [5/5] | loss: 0.864 | acc: 69.328%   == step: [150/196] [5/5] | loss: 0.874 | acc: 68.872%

   == step: [196/196] [5/5] | loss: 0.865 | acc: 69.138%
   == step: [196/196] [5/5] | loss: 0.862 | acc: 69.464%

            =======  Training Finished  ======= 


            =======  Training Finished  =======
 
"""
