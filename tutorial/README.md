<img src="https://user-images.githubusercontent.com/7837172/44953557-0fb54e80-aec9-11e8-9d38-2388bc70c5c5.png" width=15% align="right" /> 

Assume you are familiar with PyTorch, and this tutorial show you the usage of PyTorch distributed data parallel, hope my description is helpful to you.

## Code

- Single Node Single GPU Card Training [[snsc.py](snsc.py)]
- Single Node Multi-GPU Crads Training (with DataParallel) [[snmc_dp.py](snmc_dp.py)]
- Multiple Nodes Multi-GPU Cards Training (with DistributedDataParallel)
    - torch.distributed.launch [[mnmc_ddp_launch.py](mnmc_ddp_launch.py)]
    - torch.multiprocessing [[mnmc_ddp_mp.py](mnmc_ddp_mp.py)]
    - Slurm Workload Manager [[mnmc_ddp_slurm.py](mnmc_ddp_slurm.py)]
- ImageNet training example [[imagenet.py](imagenet.py)]


## Material

- **First and most important**: [PYTORCH DISTRIBUTED OVERVIEW](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Launching and configuring distributed data parallel applications](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md)
- [WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Bringing HPC Techniques to Deep Learning](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)