<div align="center">
<img src="./images/logo.png" width="300px">

**The pure and clear PyTorch Distributed Training Framework.**

</div>

* [Introduction](#introduction)
* [Requirements and Usage](#requirements-and-usage)
  * [Dependency](#dependency)
  * [Dataset](#dataset)
  * [Basic Usage](#basic-usage)
  * [Slurm Cluster Usage](#slurm-cluster-usage)
* [Baselines](#baselines)
* [Zombie processes problem](#zombie-processes-problem)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)

## Introduction


Distribuuuu is a Distributed Classification Training Framework powered by native PyTorch.

Please check [tutorial](./tutorial/) for detailed **Distributed Training** tutorials:

- Single Node Single GPU Card Training [[snsc.py](./tutorial/snsc.py)]
- Single Node Multi-GPU Cards Training (with DataParallel) [[snmc_dp.py](./tutorial/snmc_dp.py)]
- Multiple Nodes Multi-GPU Cards Training (with DistributedDataParallel)
    - torch.distributed.launch [[mnmc_ddp_launch.py](./tutorial/mnmc_ddp_launch.py)]
    - torch.multiprocessing [[mnmc_ddp_mp.py](./tutorial/mnmc_ddp_mp.py)]
    - Slurm Workload Manager [[mnmc_ddp_slurm.py](./tutorial/mnmc_ddp_slurm.py)]
- ImageNet training example [[imagenet.py](./tutorial/imagenet.py)]

For the complete training framework, please see [distribuuuu](./distribuuuu/). 

## Requirements and Usage

### Dependency

- Install **PyTorch>= 1.6** (has been tested on **1.6, 1.7.1, 1.8** and **1.8.1**)
- Install other dependencies: ``pip install -r requirements.txt``

### Dataset

Download the ImageNet dataset and move validation images to labeled subfolders, using the script [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 



<details>
  <summary>Expected datasets structure for ILSVRC</summary>

``` 
ILSVRC
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Create a directory containing symlinks:

``` bash
mkdir -p /path/to/distribuuuu/data
```

Symlink ILSVRC:

``` bash
ln -s /path/to/ILSVRC /path/to/distribuuuu/data/ILSVRC
```

</details>



### Basic Usage

Single Node with one task


``` bash
# 1 node, 8 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml
```

Distribuuuu use [yacs](https://github.com/rbgirshick/yacs), a elegant and lightweight package to define and manage system configurations.
You can setup config via a yaml file, and overwrite by other opts. If the yaml is not provided, the default configuration file will be used, please check [distribuuuu/config.py](./distribuuuu/config.py).

```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml \
    OUT_DIR /tmp \
    MODEL.SYNCBN True \
    TRAIN.BATCH_SIZE 256

# --cfg config/resnet18.yaml parse config from file
# OUT_DIR /tmp            overwrite OUT_DIR
# MODEL.SYNCBN True       overwrite MODEL.SYNCBN
# TRAIN.BATCH_SIZE 256    overwrite TRAIN.BATCH_SIZE
```


<details>
  <summary>Single Node with two tasks</summary>


```bash
# 1 node, 2 task, 4 GPUs per task (8GPUs)
# task 1:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml

# task 2:
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml
```

</details>

<details>
  <summary>Multiple Nodes Training</summary>

```bash
# 2 node, 8 GPUs per node (16GPUs)
# node 1:
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="10.198.189.10" \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml

# node 2:
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="10.198.189.10" \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml
```

</details>

### Slurm Cluster Usage

```bash
# see srun --help 
# and https://slurm.schedmd.com/ for details

# example: 64 GPUs
# batch size = 64 * 128 = 8192
# itertaion = 128k / 8192 = 156 
# lr = 64 * 0.1 = 6.4

srun --partition=openai-a100 \
     -n 64 \
     --gres=gpu:8 \
     --ntasks-per-node=8 \
     --job-name=Distribuuuu \
     python -u train_net.py --cfg config/resnet18.yaml \
     TRAIN.BATCH_SIZE 128 \
     OUT_DIR ./resnet18_8192bs \
     OPTIM.BASE_LR 6.4
```

## Baselines

**Baseline** models trained **by Distribuuuu**:

- We use SGD with momentum of 0.9, a half-period **cosine schedule**, and train for **100** epochs.
- We use a **reference learning rate** of **0.1** and a weight decay of **5e-5** (1e-5 For EfficientNet).
- The actual learning rate(**Base LR**) for each model is computed as **(batch-size / 128) * reference-lr**.
- Only standard data augmentation techniques(RandomResizedCrop and RandomHorizontalFlip) are used.

**PS: use other robust tricks(more epochs, efficient data augmentation, etc.) to get better performance.**


|                     Arch                     | Params(M) |    Total batch     | Base LR | Acc@1  | Acc@5  |                                                           model / config                                                           |
| :------------------------------------------: | :-------: | :----------------: | :-----: | :----: | :----: | :--------------------------------------------------------------------------------------------------------------------------------: |
|                   resnet18                   |  11.690   |   256 (32*8GPUs)   |   0.2   | 70.902 | 89.894 |    [Drive](https://drive.google.com/file/d/18a6QFc_DoTHo3TWkN_EsptyGmhF97sVw/view?usp=sharing) / [cfg](./config/resnet18.yaml)     |
|                   resnet18                   |  11.690   |  1024 (128*8GPUs)  |   0.8   | 70.994 | 89.892 |                                                                                                                                    |
|                   resnet18                   |  11.690   | 8192 (128*64GPUs)  |   6.4   | 70.165 | 89.374 |                                                                                                                                    |
|                   resnet18                   |  11.690   | 16384 (256*64GPUs) |  12.8   | 68.766 | 88.381 |                                                                                                                                    |
|               efficientnet_b0                |   5.289   |   512 (64*8GPUs)   |   0.4   | 74.540 | 91.744 | [Drive](https://drive.google.com/file/d/1nSLQBBRKnAJYdoFhUUVsV8qI5270ooq3/view?usp=sharing) / [cfg](./config/efficientnet_b0.yaml) |
|                   resnet50                   |  25.557   |   256 (32*8GPUs)   |   0.2   | 77.252 | 93.430 |    [Drive](https://drive.google.com/file/d/1rUY1mSYTxe7jWzzcWrreg398tbSNXtnv/view?usp=sharing) / [cfg](./config/resnet50.yaml)     |
| [botnet50](https://arxiv.org/abs/2101.11605) |  20.859   |   256 (32*8GPUs)   |   0.2   | 77.604 | 93.682 |    [Drive](https://drive.google.com/file/d/1-jvhJaMyy-KziAuFnmt5rkoZrm5364UF/view?usp=sharing) / [cfg](./config/botnet50.yaml)     |
|                 regnetx_160                  |  54.279   |   512 (64*8GPUs)   |   0.4   | 79.992 | 95.118 |   [Drive](https://drive.google.com/file/d/1w2LtMKiLHwz27fJOmymQmPCX1yPDuPsm/view?usp=sharing) / [cfg](./config/regnetx_160.yaml)   |
|                 regnety_160                  |  83.590   |   512 (64*8GPUs)   |   0.4   | 80.598 | 95.090 |   [Drive](https://drive.google.com/file/d/1dmD94jeZCaYLI9DhbMN0V1uG6_KHkx_o/view?usp=sharing) / [cfg](./config/regnety_160.yaml)   |
|                 regnety_320                  |  145.047  |   512 (64*8GPUs)   |   0.4   | 80.824 | 95.276 |   [Drive](https://drive.google.com/file/d/1pVbSy4YSlWBra1C2NLTNwJkk_zOTomZg/view?usp=sharing) / [cfg](./config/regnety_320.yaml)   |
## Zombie processes problem


Before PyTorch1.8, ``torch.distributed.launch`` will leave some zombie processes after using  ``Ctrl`` + ``C``, try to use the following cmd to kill the zombie processes. ([fairseq/issues/487](https://github.com/pytorch/fairseq/issues/487)):

```bash
kill $(ps aux | grep YOUR_SCRIPT.py | grep -v grep | awk '{print $2}')
```

PyTorch >= 1.8 is suggested, which fixed the issue about zombie process. ([pytorch/pull/49305](https://github.com/pytorch/pytorch/pull/49305))


## Acknowledgments

Provided codes were adapted from:

- [facebookresearch/pycls](https://github.com/facebookresearch/pycls)
- [pytorch/examples](https://github.com/pytorch/examples/)
- [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)

I strongly recommend you to choose [pycls](https://github.com/facebookresearch/pycls), a brilliant image classification codebase and adopted by a number of projects at [Facebook AI Research](https://github.com/facebookresearch).



## Citation

```
@misc{bigballon2021distribuuuu,
  author = {Wei Li},
  title = {Distribuuuu: The pure and clear PyTorch Distributed Training Framework},
  howpublished = {\url{https://github.com/BIGBALLON/distribuuuu}},
  year = {2021}
}
```

Feel free to contact me if you have any suggestions or questions, issues are welcome,
create a PR if you find any bugs or you want to contribute. :cake:
