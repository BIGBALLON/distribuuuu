import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from iopath.common.file_io import g_pathmgr
from loguru import logger

import distribuuuu.config as config
from distribuuuu.config import cfg


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
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
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29566"
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


def setup_seed(rank):
    """Sets up environment for training or testing."""
    if rank == 0:
        g_pathmgr.mkdirs(cfg.OUT_DIR)
        config.dump_cfg()

    if cfg.RNG_SEED:
        np.random.seed(cfg.RNG_SEED + rank)
        torch.manual_seed(cfg.RNG_SEED + rank)
        random.seed(cfg.RNG_SEED + rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC


def setup_logger(rank, local_rank):
    logger.remove()
    fmt_str = "[{time:YYYY-MM-DD HH:mm:ss}] {message}"
    if rank == 0:
        logger.add(
            f"{cfg.OUT_DIR}/{time.time()}.log",
            format=fmt_str,
        )
    logger.add(sys.stderr, format=fmt_str)
    logger.debug(f"LOCAL_RANK: {local_rank}, RANK: {rank}")
    if rank == 0:
        logger.debug(f"\n{cfg.dump()}")


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group.
    """
    # There is no need for reduction in the single-proc case
    gpus = torch.distributed.get_world_size()
    if gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / gpus)
    return tensors


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length, size):
        self.len = length
        self.data = torch.randn([length] + size)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return self.len


def construct_train_loader():
    """Constructs the train data loader for ILSVRC dataset."""
    traindir = os.path.join(cfg.TRAIN.DATASET, cfg.TRAIN.SPLIT)
    if cfg.MODEL.DUMMY_INPUT:
        trainset = DummyDataset(1000, [3, 224, 224])
    else:
        trainset = torchvision.datasets.ImageFolder(
            root=traindir,
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(cfg.TRAIN.IM_SIZE),
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
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        sampler=train_sampler,
        drop_last=True,
    )
    return train_loader


def construct_val_loader():
    """Constructs the validate data loader for ILSVRC dataset."""
    valdir = os.path.join(cfg.TRAIN.DATASET, cfg.TEST.SPLIT)
    if cfg.MODEL.DUMMY_INPUT:
        valset = DummyDataset(1000, [3, 224, 224])
    else:
        valset = torchvision.datasets.ImageFolder(
            root=valdir,
            transform=transforms.Compose(
                [
                    transforms.Resize(cfg.TEST.IM_SIZE),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
    val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.TRAIN.WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        drop_last=False,
    )
    return val_loader


def construct_optimizer(model):
    """Constructs the optimizer."""
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        dampening=cfg.OPTIM.DAMPENING,
        nesterov=cfg.OPTIM.NESTEROV,
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display training progress"""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.time_eta = None

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.time_eta:
            logger.info(f"ETA: {self.time_eta / 3600:.2f}h " + " | ".join(entries))
        else:
            logger.info(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def cal_eta(self, iters, total_iter, tic=time.time(), cur_epoch=0, start_epoch=0):
        time_elapsed = time.time() - tic
        ratio_running = (
            cur_epoch - start_epoch + iters / total_iter
        ) / cfg.OPTIM.MAX_EPOCH
        ratio_remaining = 1 - (cur_epoch + iters / total_iter) / cfg.OPTIM.MAX_EPOCH
        self.time_eta = time_elapsed / ratio_running * ratio_remaining


def construct_meters():
    """Constructs the meters."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":5.3f")
    losses = AverageMeter("Loss", ":6.4f")
    top1 = AverageMeter("Acc@1", ":6.3f")
    topk = AverageMeter(f"Acc@{cfg.TRAIN.TOPK}", ":6.3f")
    return batch_time, data_time, losses, top1, topk


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def lr_fun_steps(cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULT ** ind


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.OPTIM.MAX_EPOCH))
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.OPTIM.LR_POLICY
    err_str = "exp lr policy requires OPTIM.MIN_LR to be greater than 0."
    assert cfg.OPTIM.LR_POLICY != "exp" or cfg.OPTIM.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun()(cur_epoch) * cfg.OPTIM.BASE_LR
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# Common prefix for checkpoint file names
_NAME_PREFIX = "ckpt_ep_"

# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = f"{_NAME_PREFIX}{epoch:03d}.pth.tar"
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [f for f in g_pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not g_pathmgr.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in g_pathmgr.ls(checkpoint_dir))


def count_parameters(model):
    parms = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb_size = parms * 4.0 / 1024 / 1024
    parms = parms / 1000000
    return f"Params(M): {parms:.3f}, Model Size(MB): {mb_size:.3f}"


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


def save_checkpoint(model, optimizer, epoch, best_acc1, best):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if torch.distributed.get_rank() != 0:
        return
    # Ensure that the checkpoint dir exists
    g_pathmgr.mkdirs(get_checkpoint_dir())
    # Record the state
    state_dict = unwrap_model(model).state_dict()
    checkpoint = {
        "epoch": epoch,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "best_acc1": best_acc1,
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)

    # If best, save the weight only
    if best:
        torch.save(state_dict, os.path.join(cfg.OUT_DIR, "best.pth.tar"))
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = f"CHECKPOINT '{checkpoint_file}' NOT FOUND"
    assert g_pathmgr.exists(checkpoint_file), err_str
    start_epoch = best_acc1 = 0
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if "state_dict" in checkpoint:
        unwrap_model(model).load_state_dict(checkpoint["state_dict"])
        if optimizer:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                start_epoch = checkpoint["epoch"] + 1
                best_acc1 = checkpoint["best_acc1"]
            except BaseException:
                logger.info(f"CAN'T FOUND OPTIMIZER in {checkpoint_file}")
    else:
        unwrap_model(model).load_state_dict(checkpoint)
    if torch.distributed.get_rank() == 0:
        logger.info(f"LOADED '{checkpoint_file}'")
    return start_epoch, best_acc1
