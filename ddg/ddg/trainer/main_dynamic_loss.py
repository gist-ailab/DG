import json
import argparse
from collections import OrderedDict
from ddg.trainer import Trainer
from ddg.trainer import LRD_Trainer
from ddg.trainer import DDG_Trainer_Loss

if __name__ == '__main__':
    DDG_Trainer_Loss().run()
