import torch.nn as nn
from enum import Enum
from typing import List
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
from torch import Tensor
from logger_tt import logger
from collections import OrderedDict
import sklearn.metrics as skmet
# from terminaltables import SingleTable
# from termcolor import colored

__all__ = ['Summary', 'AverageMeter', 'ProgressMeter', 'load_state_dict', 'matplotlib_imshow', 'images_to_probs', 'plot_classes_preds']

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    학습된 신경망과 이미지 목록으로부터 예측 결과 및 확률을 생성합니다
    '''
    output = images
    for model in net:
        output = net[model](output) 
    # output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
    Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
    예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
    함수를 사용합니다.
    '''
    # classes=['0','1','2', '3', '4', '5','6']
    preds, probs = images_to_probs(net, images)
    # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch + 1)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def on_load_checkpoint(self, checkpoint: dict) -> None:
    state_dict = checkpoint
    model_state_dict = self.state_dict()
    is_changed = False
    ms_keys = []
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.info(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                ms_keys.append(k)
                state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            logger.info(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        checkpoint.pop("optimizer_states", None)
    return ms_keys

def load_state_dict(model: nn.Module, state_dict: 'OrderedDict[str, Tensor]',
                    allowed_missing_keys: List = None):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Args:
        model (torch.nn.Module): a torch.nn.Module object where state_dict load for.
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        allowed_missing_keys (List, optional): not raise `RuntimeError` if missing_keys
        equal to allowed_missing_keys.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Note:
        If a parameter or buffer is registered as ``None`` and its corresponding key
        exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
        ``RuntimeError``.
    """
    # missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False) #strict=allowed_missing_keys is None)
    ms_keys = on_load_checkpoint(model, state_dict) #strict=allowed_missing_keys is None)
    
    loaded_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    successfully_loaded_keys = loaded_keys & checkpoint_keys
    return ms_keys
    msgs: List[str] = []
    # raise_error = False
    # if len(unexpected_keys) > 0:
        # raise_error = True
    #     msgs.insert(
    #         0, 'Unexpected key(s) in state_dict: {}. '.format(
    #             ', '.join('"{}"'.format(k) for k in unexpected_keys)))
    # if len(missing_keys) > 0:
    #     # if allowed_missing_keys is None or sorted(missing_keys) != sorted(allowed_missing_keys):
    #         # raise_error = True
    #     msgs.insert(
    #         0, 'Missing key(s) in state_dict: {}. '.format(
    #             ', '.join('"{}"'.format(k) for k in missing_keys)))
    # if raise_error:
    #     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #         model.__class__.__name__, "\n\t".join(msgs)))
        
    # msgs.insert(
    #     0, 'loaded key(s) in state_dict: {}'.format(
    #     ', '.join('"{}"'.format(k) for k in successfully_loaded_keys)))
    # if len(msgs) > 0:
    #     logger.info('\nInfo(s) in loading state_dict for {}:\n\t{}'.format(
    #                     model.__class__.__name__, "\n\t".join(msgs)))


    # return missing_keys, unexpected_keys