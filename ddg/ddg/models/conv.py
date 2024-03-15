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
import numpy as np
from ddg.utils.fb import calculate_FB_bases


__all__ = ['Conv2d_LRD_Dynamic', 'Conv2dDynamic', 'Conv2d_LRD', '_ConvNd_LRD']

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
     

        self.common_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank], dtype=torch.float), requires_grad=True)
        self.specific_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.num_domain], dtype=torch.float), requires_grad=True)
            
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


class Conv2d_LRD(_ConvNd_LRD): #_`ConvNd_LRD`):
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

    # conv weight [Kernel^2, in_Channel, Out_channel]
    # bias [Out_channel]
    def forward(self, x, dynamic_param):
        feature = self._conv_forward(x, self.weight, self.bias)
        #             [D] [num_domain,  rank] @ [Rank, num_Domain]
        self.specific_term = self.specific_weight @ dynamic_param 
        #    
        self.common_conv_weight = self.common_weight @ self.weight
        self.specific_conv_weight = self.specific @ self.weight

        self.common_conv_bias = self.bias @ self.common_weight
        self.specific_conv_bias = self.b @ self.specific_term
#   
        specific_feature = x @ self.specific_conv_weight + self.specific_conv_bias
        common_feature = x @ self.common_conv_weight + self.common_conv_bias
         
        return feature, common_feature, specific_feature   
    
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            feature = F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        feature = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        return feature
# #%%
# import torch
# tmp = torch.nn.Parameter(torch.normal(mean=0., std=1e-4, size=[], dtype=torch.float), requires_grad=True)
# tmp.shape
# c_wts = torch.randn((256, 2))
# c_wts = torch.cat((torch.ones((256, 1), dtype=torch.float)*tmp, c_wts), 1)
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
        x = torch.tensor(x, dtype=torch.float32)
        attention_x = x if attention_x is None else attention_x
        # print(f'attention_x shape: {attention_x.shape}')
        y = self.attention(attention_x)
        # print(f'y shape: {y.shape}')
        out = self.conv(x)
        # print(f'out shape: {out.shape}')
        for i, template in enumerate(self.kernel_templates):
            out = out + self.kernel_templates[template](x) * y[:, i].view(-1, 1, 1, 1)
            # print(f'out shape: {out.shape}')
        # print(f'final out shape: {out.shape}')

        return out


class Conv2d_LRD_Dynamic(Conv2dDynamic):
    r"""Pytorch implementation for 2D DCF Convolution operation.
    Link to ICML paper:
    https://arxiv.org/pdf/1802.04145.pdf


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, optional): Zero-padding added to both sides of
            the input. Default: 0
        num_bases (int, optional): Number of basis elements for decomposition.
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        mode (optional): Either `mode0` for two-conv or `mode1` for reconstruction + conv.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 * \text{padding}[0] - \text{dilation}[0]
                        * (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 * \text{padding}[1] - \text{dilation}[1]
                        * (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::
        
        >>> from DCF import *
        >>> m = Conv_DCF(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    """
    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
        num_bases=-1, bias=True, bases_grad=False, dilation=1, initializer='FB', mode='mode1', squeeze:int=None, attention_in_channels:int=None) -> None:
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


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_bases = num_bases
        self.stride = stride
        self.padding = padding
        
        self.kernel_list = {}
        assert mode in ['mode0', 'mode1'], 'Only mode0 and mode1 are available at this moment.'
        self.mode = mode
        self.bases_grad = bases_grad
        self.dilation = dilation

        assert initializer in ['FB', 'random'], 'Initializer should be either FB or random, other methods are not implemented yet'

        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise Exception('Kernel size for FB initialization only supports odd number for now.')
            base_np, _, _ = calculate_FB_bases(int((kernel_size-1)/2))
            if num_bases > base_np.shape[1]:
                raise Exception('The maximum number of bases for kernel size = %d is %d' %(kernel_size, base_np.shape[1]))
            elif num_bases == -1:
                num_bases = base_np.shape[1]
                common_num_base = 2
                idx_list = np.arange(num_bases)
                common_idx = np.random.choice(idx_list, common_num_base, replace=False)
                
                common_base_np = base_np[:, common_idx]
                specific_idx = np.delete(idx_list, common_idx)
                specific_base_np = base_np[:, specific_idx]
            else:
                base_np = base_np[:, :num_bases]
            # base_np = base_np.reshape(kernel_size, kernel_size, num_bases)
            # base_np = np.array(np.expand_dims(base_np.transpose(2,0,1), 1), np.float32)
            
            common_base_np = common_base_np.reshape(kernel_size, kernel_size, common_num_base)
            common_base_np = torch.tensor(np.expand_dims(common_base_np.transpose(2,0,1), 1), dtype=torch.float32)
            specific_base_np = specific_base_np.reshape(kernel_size, kernel_size, num_bases-common_num_base)
            specific_base_np = torch.tensor(np.expand_dims(specific_base_np.transpose(2,0,1), 1), dtype=torch.float32)

        else:
            if num_bases <= 0:
                raise Exception('Number of basis elements must be positive when initialized randomly.')
            base_np = np.random.randn(num_bases, 1, kernel_size, kernel_size)

        if bases_grad:
            self.common_bases = Parameter(torch.tensor(common_base_np, dtype=torch.float32), requires_grad=bases_grad)
            self.common_bases.data.normal_(0, 1.0)

            self.specific_bases = Parameter(torch.tensor(specific_base_np, dtype=torch.float32), requires_grad=bases_grad)
            self.specific_bases.data.normal_(0, 1.0)
            # self.bases.data.uniform_(-1, 1)
        else:
            self.register_buffer('common_bases', torch.tensor(common_base_np, requires_grad=False, dtype=torch.float32).float())
            self.register_buffer('specific_bases', torch.tensor(specific_base_np, requires_grad=False, dtype=torch.float32).float())
        
        self.common_weight = Parameter(torch.tensor(torch.Tensor(
                out_channels, in_channels*common_num_base, 1, 1), dtype=torch.float32))
        
        self.specific_weight = Parameter(torch.tensor(torch.Tensor(
                        out_channels, in_channels*(num_bases-common_num_base), 1, 1), dtype=torch.float32))        
        # print(bias)
        
        if bias:
            self.common_bias = Parameter(torch.tensor(torch.Tensor(out_channels), dtype=torch.float32))
            self.specific_bias = Parameter(torch.tensor(torch.Tensor(out_channels), dtype=torch.float32))
        else:
            self.register_parameter('common_bias', None)
            self.register_parameter('specific_bias', None)
            self.reset_parameters()

        self.num_bases = num_bases
        if self.mode == 'mode1':
            self.common_weight.data = self.common_weight.data.view(out_channels, in_channels, common_num_base)
            self.common_bases.data = self.common_bases.data.view(common_num_base, kernel_size, kernel_size)
            self.specific_weight.data = self.specific_weight.data.view(out_channels, in_channels, num_bases-common_num_base)
            self.specific_bases.data = self.specific_bases.data.view(num_bases-common_num_base, kernel_size, kernel_size)
            
            # self.forward = self.forward_mode1
        # else:
        #     self.forward = self.forward_mode0
        
        self.common_num_base = torch.tensor(common_num_base, dtype=torch.float32)
        # self.common_weight = torch.tensor(self.common_weight, dtype=torch.float32)
        # self.specific_weight = torch.tensor(self.specific_weight, dtype=torch.float32)
        # self.common_bias = torch.tensor(self.common_bias, dtype=torch.float32)
        # self.specific_bias = torch.tensor(self.specific_bias, dtype=torch.float32)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.common_weight.size(1))
        stdv = 1. / math.sqrt(self.specific_weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.common_weight.data.normal_(0, stdv) #Normal works better, working on more robust initializations
        if self.common_bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.common_bias.data.zero_()
        self.specific_weight.data.normal_(0, stdv) #Normal works better, working on more robust initializations
        if self.specific_bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.specific_bias.data.zero_()

    def forward(self, x, attention_x=None):
        x = torch.tensor(x, dtype=torch.float32)
        if self.mode == 'mode1':
            common_feature, specific_feature = self.forward_imply_mode1(x)
        else: common_feature, specific_feature = self.forward_imply_mode0(x)
        
        attention_x = x if attention_x is None else attention_x
        y = self.attention(attention_x)
        for i, template in enumerate(self.kernel_templates):
            specific_feature = specific_feature + self.kernel_templates[template](specific_feature) * y[:, i].view(-1, 1, 1, 1)
        # print(f'common_{common_feature.shape}')
        # print(f'specific_{specific_feature.shape}')
        feature = torch.cat([common_feature, specific_feature], 2)
        return feature, common_feature, specific_feature
    

    def forward_imply_mode0(self, x):
     
        N, C, H, W = x.size()
        feature_list = []
        x = x.view(N*C, 1, H, W)
        
        common_feature = F.conv2d(x, self.common_bases,
            None, self.stride, self.padding, dilation=self.dilation)
        common_feature = torch.tensor(common_feature, dtype=torch.float32)
        specific_feature = F.conv2d(x, self.specific_bases,
            None, self.stride, self.padding, dilation=self.dilation)
        specific_feature = torch.tensor(specific_feature, dtype=torch.float32)
        
        H = int((H-self.kernel_size+2*self.padding)/self.stride+1)
        W = int((W-self.kernel_size+2*self.padding)/self.stride+1)

        common_feature = common_feature.view(N, C*int(self.common_num_base), H, W)
        specific_feature = specific_feature.view(N, C*(self.num_bases-int(self.common_num_base)), H, W)


        common_feature_out = F.conv2d(common_feature, self.common_weight, self.common_bias, 1, 0)
        specific_feature_out = F.conv2d(specific_feature, self.specific_weight, self.specific_bias, 1, 0)
        return common_feature_out, specific_feature_out

    def forward_imply_mode1(self, x):
        common_rec_kernel = torch.einsum('abc,cdf->abdf', self.common_weight, self.common_bases)
        specific_rec_kernel = torch.einsum('abc,cdf->abdf', self.specific_weight, self.specific_bases)
        common_feature = F.conv2d(x, common_rec_kernel,
            self.common_bias, self.stride, self.padding, dilation=self.dilation)
        specific_feature = F.conv2d(x, specific_rec_kernel,
            self.specific_bias, self.stride, self.padding, dilation=self.dilation)
        return common_feature, specific_feature

    def extra_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}, num_bases={num_bases}' \
            ', bases_grad={bases_grad}, mode={mode}'.format(**self.__dict__)


# if __name__ == '__main__':
#     conv = Conv_DCF(10, 20, 3)
#     conv2 = Conv_DCF(10, 20, 3, mode='mode0')
#     data = torch.randn(2, 10, 16, 16)
#     print(conv(data).shape)
#     print(conv2(data).shape)
# class Conv2dLRDynamic(Conv2dDynamic):
#     def __init__(self, in_channels: int,
#                  out_channels: int,
#                  kernel_size: int,
#                  stride: int,
#                  padding: int,
#                  bias: bool = True,
#                  squeeze: int = None,
#                  attention_in_channels: int = None) -> None:
#         super().__init__()

#         if kernel_size // 2 != padding:
#             # Only when this condition is met, we can ensure that different
#             # kernel_size can obtain feature maps of consistent size.
#             # Let I, K, S, P, O: O = (I + 2P - K) // S + 1, if P = K // 2, then O = (I - K % 2) // S + 1
#             # This means that the output of two different Ks with the same parity can be made the same by adjusting P.
#             raise ValueError('`padding` must be equal to `kernel_size // 2`.')
#         if kernel_size % 2 == 0:
#             raise ValueError('Kernel_size must be odd now because the templates we used are odd (kernel_size=1).')

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                               padding=padding, bias=bias)
#         self.kernel_templates = nn.ModuleDict()
#         self.kernel_templates['conv_nn'] = nn.Conv2d(in_channels, out_channels,
#                                                      kernel_size=kernel_size,
#                                                      stride=stride,
#                                                      padding=padding,
#                                                      groups=min(in_channels, out_channels),
#                                                      bias=bias)
#         self.kernel_templates['conv_11'] = nn.Conv2d(in_channels, out_channels,
#                                                      kernel_size=1,
#                                                      stride=stride,
#                                                      padding=0,
#                                                      bias=bias)
#         self.kernel_templates['conv_n1'] = nn.Conv2d(in_channels, out_channels,
#                                                      kernel_size=(kernel_size, 1),
#                                                      stride=stride,
#                                                      padding=(padding, 0),
#                                                      bias=bias)
#         self.kernel_templates['conv_1n'] = nn.Conv2d(in_channels, out_channels,
#                                                      kernel_size=(1, kernel_size),
#                                                      stride=stride,
#                                                      padding=(0, padding),
#                                                      bias=bias)
#         self.attention = Attention(attention_in_channels if attention_in_channels else in_channels,
#                                    4, squeeze, bias=bias)

#     def forward(self, x, attention_x=None):
#         attention_x = x if attention_x is None else attention_x
#         # print(f'attention_x shape: {attention_x.shape}')
#         y = self.attention(attention_x)
#         # print(f'y shape: {y.shape}')
#         out = self.conv(x)
#         # print(f'out shape: {out.shape}')
#         for i, template in enumerate(self.kernel_templates):
#             out += self.kernel_templates[template](x) * y[:, i].view(-1, 1, 1, 1)
        
#         return specified_out, common_out 



