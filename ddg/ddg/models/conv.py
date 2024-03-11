import torch.nn as nn
from .attention import Attention
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import math


__all__ = ['Conv2dDynamic', 'Conv2d_LRD', '_ConvNd_LRD']

class Conv2d_LRD(_ConvNd_LRD):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
      
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    
    def forward(self, x, dynamic_param):
        feature = self._conv_forward(x, self.weight, self.bias)

        self.specific_term = self.specific_weight @ dynamic_param 
        self.common_conv_weight = self.common_weight @ self.weight
        self.specific_conv_weight = self.specific @ self.weight

        self.common_conv_bias = self.bias @ self.common_weight
        self.specific_conv_bias = self.b @ self.specific_term
#   
        specific_feature = input @ self.specific_conv_weight + self.specific_conv_bias
        common_feature = input @ self.common_conv_weight + self.common_conv_bias
         
        return feature, common_feature, specific_feature   
    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            feature = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        feature = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return feature


class Conv2dDynamic(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = True,
                 squeeze: int = None,
                 attention_in_channels: int = None) -> None:
        super(Conv2dDynamic, self).__init__()

        if kernel_size // 2 != padding:
            # Only when this condition is met, we can ensure that different
            # kernel_size can obtain feature maps of consistent size.
            # Let I, K, S, P, O: O = (I + 2P - K) // S + 1, if P = K // 2, then O = (I - K % 2) // S + 1
            # This means that the output of two different Ks with the same parity can be made the same by adjusting P.
            raise ValueError('`padding` must be equal to `kernel_size // 2`.')
        if kernel_size % 2 == 0:
            raise ValueError('Kernel_size must be odd now because the templates we used are odd (kernel_size=1).')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.kernel_templates = nn.ModuleDict()
        self.kernel_templates['conv_nn'] = nn.Conv2d(in_channels, out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     padding=padding,
                                                     groups=min(in_channels, out_channels),
                                                     bias=bias)
        self.kernel_templates['conv_11'] = nn.Conv2d(in_channels, out_channels,
                                                     kernel_size=1,
                                                     stride=stride,
                                                     padding=0,
                                                     bias=bias)
        self.kernel_templates['conv_n1'] = nn.Conv2d(in_channels, out_channels,
                                                     kernel_size=(kernel_size, 1),
                                                     stride=stride,
                                                     padding=(padding, 0),
                                                     bias=bias)
        self.kernel_templates['conv_1n'] = nn.Conv2d(in_channels, out_channels,
                                                     kernel_size=(1, kernel_size),
                                                     stride=stride,
                                                     padding=(0, padding),
                                                     bias=bias)
        self.attention = Attention(attention_in_channels if attention_in_channels else in_channels,
                                   4, squeeze, bias=bias)

    def forward(self, x, attention_x=None):
        attention_x = x if attention_x is None else attention_x
        print(f'attention_x shape: {attention_x.shape}')
        y = self.attention(attention_x)
        print(f'y shape: {y.shape}')
        out = self.conv(x)
        print(f'out shape: {out.shape}')
        for i, template in enumerate(self.kernel_templates):
            out += self.kernel_templates[template](x) * y[:, i].view(-1, 1, 1, 1)
            print(f'out shape: {out.shape}')
        print(f'final out shape: {out.shape}')

        return out


class _ConvNd_LRD(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None, rank=2) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.rank = rank
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
##########################################################################################
# 수정
##########################################################################################
        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
#########################################################################################
     

        self.common_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank], dtype=torch.float, device='cuda'), requires_grad=True)
        self.specific_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, 2], dtype=torch.float, device='cuda'), requires_grad=True)
            
            # self.embedding_matrix = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank], dtype=torch.float, device='cuda'), requires_grad=True)
            # self.embedding_matrix = self.dynamic_param(torch.normail())  #[rank]
        

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
