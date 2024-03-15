import torch
import torch.distributed as dist
import traceback
import os
import sys

# class Partition(object):

#     def __init__(self, data, index):
#         self.data = data
#         self.index = index

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]


# class DataPartitioner(object):

#     def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
#         self.data = data
#         self.partitions = []
#         rng = Random()
#         rng.seed(seed)
#         data_len = len(data)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)

#         for frac in sizes:
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]

#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])


def cleanup():
    dist.destroy_process_group()
    print('=> Clean up process group')


def init_for_distributed(opts):

    # 1. setting for distributed training

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opts.rank = int(os.environ["RANK"])
        opts.world_size = int(os.environ['WORLD_SIZE'])
        
        opts.local_rank = int(os.environ['LOCAL_RANK'])

        # logger.info(f'Using RANK={opts.rank}, WORLD_SIZE={opts.world_size}, LOCAL_RANK={opts.local_rank}')
    # elif 'SLURM_PROCID' in os.environ:
    #     opts.rank = int(os.environ['SLURM_PROCID'])
    #     opts.gpu = opts.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        opts.distributed = False
        return
    
    torch.cuda.set_device(opts.local_rank)
    opts.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        opts.rank, 'env://'), flush=True)
    opts.dist_url = f'env://:{opts.port}'
    torch.distributed.init_process_group(backend=opts.dist_backend, init_method=opts.dist_url,
                                         world_size=opts.world_size, rank=opts.rank)
    # torch.cuda.set_device(opts.gpu)
    
    torch.distributed.barrier()
    # if dist.get_rank() == 0:
    #     setup_for_distributed()

    # if torch.distributed.get_rank() == 0:
    #     print('=> world_size={}, rank={}, backend={}, gpu={}'.format(
    #         opts.world_size, opts.rank, opts.dist_backend, opts.gpu), flush=True)
 


    
    
def setup_for_distributed():
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    import traceback
    import sys
    builtin_print = __builtin__.print
    excepthook = sys.excepthook
    
    def is_master():
        return dist.get_rank() == 0

    def print(*args, **kwargs):
        # force = kwargs.pop('force', False)
        if is_master(): #or force:
            builtin_print(*args, **kwargs)

    def except_hook(type, value, tb):
        if is_master():
            traceback.print_excepthook(type, value, tb)
        else: pass
           # def logger(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if is_master or force:
    #         builtin_print(*args, **kwargs)
    sys.excepthook = except_hook
    __builtin__.print = print
    print(f'===> Only master process will print')
            

    
