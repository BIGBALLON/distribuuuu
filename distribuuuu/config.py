import argparse

from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CN

_C = CN()
cfg = _C

_C.MODEL = CN()
_C.MODEL.ARCH = "resnet50"
_C.MODEL.PRETRAINED = False
_C.MODEL.SYNCBN = True
_C.MODEL.WEIGHTS = None

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.IM_SIZE = 224
_C.TRAIN.DATASET = "./data/ILSVRC/"
_C.TRAIN.SPLIT = "train"

_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.LOAD_OPT = True


_C.TRAIN.WORKERS = 4
_C.TRAIN.PIN_MEMORY = True
_C.TRAIN.PRINT_FREQ = 30
_C.TRAIN.TOPK = 5

_C.TEST = CN()
_C.TEST.DATASET = "./data/ILSVRC/"
_C.TEST.SPLIT = "val"
_C.TEST.BATCH_SIZE = 128
_C.TEST.IM_SIZE = 256
_C.TEST.PRINT_FREQ = 10

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True

_C.OPTIM = CN()
# Learning rate policy select from {'cos', 'steps'}
_C.OPTIM.MAX_EPOCH = 100
_C.OPTIM.LR_POLICY = "cos"
_C.OPTIM.BASE_LR = 0.2
_C.OPTIM.MIN_LR = 0.0
_C.OPTIM.STEPS = []
_C.OPTIM.LR_MULT = 0.1
_C.OPTIM.MOMENTUM = 0.9
_C.OPTIM.DAMPENING = 0.0
_C.OPTIM.NESTEROV = True
_C.OPTIM.WARMUP_FACTOR = 0.1
_C.OPTIM.WARMUP_EPOCHS = 5
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Output directory
_C.OUT_DIR = "./exp"

_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, default=None, type=str)
    help_s = "LOCAL_RANK for torch.distributed.launch.(see --use_env for more details)"
    parser.add_argument("--local_rank", help=help_s, default=None)
    help_s = "See distribuuuu/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cfg_file is not None:
        merge_from_file(args.cfg_file)
    if args.opts is not None:
        _C.merge_from_list(args.opts)
