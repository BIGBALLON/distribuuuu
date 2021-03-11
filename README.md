<div align="center">
<img src="./images/logo.png" width="300px">

**The pure and clear PyTorch Distributed Training Framework.**

</div>


## Introduction


Distribuuuu is a Distributed Classification Training Framework powered by native PyTorch.

Please check [tutorial](./tutorial/) for detailed **Distributed Training** tutorials:

- Single Node Single GPU Card Training [[snmc_dp.py](./tutorial/snsc.py)]
- Single Node Multi-GPU Crads Training (with DataParallel) [[snmc_dp.py](./tutorial/snmc_dp.py)]
- Multiple Nodes Multi-GPU Cards Training (with DistributedDataParallel)
    - torch.distributed.launch [[mnmc_ddp_launch.py](./tutorial/mnmc_ddp_launch.py)]
    - torch.multiprocessing [[mnmc_ddp_mp.py](./tutorial/mnmc_ddp_mp.py)]
    - Slurm Workload Manager [[mnmc_ddp_slurm.py](./tutorial/mnmc_ddp_slurm.py)]
- ImageNet training example [[imagenet.py](./tutorial/imagenet.py)]

For the complete training framework, please see [distribuuuu](./distribuuuu/). 

## Requirements and Usage

### Dependency

- Install **PyTorch>= 1.5** (has been tested on **1.5, 1.7.1** and **1.8**)
- Install other dependencies: ``pip install -r requirements.txt``


### Dataset

Download the ImageNet dataset and move validation images to labeled subfolders, using the script [valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 

<details>
  <summary>Expected datasets structure for ILSVRC</summary>

``` bash
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



### Local Machine Usage

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

Distribuuuu use [yacs](https://github.com/rbgirshick/yacs), a elegant package to define and manage system configurations.
You can setup config via a yaml file, and overwrite by other opts:

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
# batch size = 64*128 = 
# itertaion = 128k / 8192 -> 157 

srun --partition=openai-a100 \
     -n 64 \
     --gres=gpu:8 \
     --ntasks-per-node=8 \
     --job-name=Distribuuuu \
     python -u train_net.py --cfg config/resnet18.yaml
```

## Acknowledgments

Provided codes were adapted from:

- [facebookresearch/pycls](https://github.com/facebookresearch/pycls)
- [pytorch/examples](https://github.com/pytorch/examples/)
- [open-mmlab/mmcv](https://github.com/open-mmlab/mmcv)

I strongly recommend you to choose [pycls](https://github.com/facebookresearch/pycls), a brilliant codebase and adopted by a number of projects at [Facebook AI Research](https://github.com/facebookresearch).


Feel free to contact me if you have any suggestions or questions, issues are welcome,
create a PR if you find any bugs or you want to contribute. :cake: