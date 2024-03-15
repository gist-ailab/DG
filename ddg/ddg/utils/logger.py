import torch.distributed as dist
import logging

class DistributedLogger:
    def __init__(self, logger):
        self.logger = logger
    def info(self, msg):
        if dist.get_rank() == 0:
            self.logger.info(msg)
        # self.logger.info(msg)