import json
import argparse
from collections import OrderedDict
from ddg.trainer import Trainer
from ddg.trainer import LRD_Trainer

if __name__ == '__main__':
    LRD_Trainer().run()
    # lrd.run_progress(lrd.run(), lrd.args.world_size)
    # lrd.cleanup()

