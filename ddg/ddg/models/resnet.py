import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from ddg.utils import load_state_dict
from ddg.utils import MODELS_REGISTRY
from .conv import Conv2dDynamic
from .conv import Conv2d_LRD
from .conv import Conv2d_LRD_Dynamic
from ..methods.mixstyle import MixStyle

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50',
          'resnet50_dynamic', 'resnet50_dynamic', 'resnet50_dynamic',
          'resnet18_dynamic_lrd', 'resnet34_dynamic_lrd', 'resnet50_dynamic_lrd',]

        #    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
        #    'wide_resnet50_2', 'wide_resnet101_2',
        #    'resnet18_backbone', 'resnet34_backbone', 'resnet50_backbone', 'resnet101_backbone',
        #    'resnet152_backbone', 'resnext50_32x4d_backbone', 'resnext101_32x8d_backbone',
        #    'wide_resnet50_2_backbone', 'wide_resnet101_2_backbone',
        #    'resnet18_dynamic', 'resnet50_dynamic', 'resnet101_dynamic',
        #    'resnet18_dynamic_backbone', 'resnet50_dynamic_backbone', 'resnet101_dynamic_backbone',
        #    'resnet18_ms_l123', 'resnet18_ms_l12', 'resnet18_ms_l1',
        #    'resnet50_ms_l123', 'resnet50_ms_l12', 'resnet50_ms_l1',
        #    'resnet101_ms_l123', 'resnet101_ms_l12', 'resnet101_ms_l1',
        #    'resnet18_ms_l123_backbone', 'resnet18_ms_l12_backbone', 'resnet18_ms_l1_backbone',
        #    'resnet50_ms_l123_backbone', 'resnet50_ms_l12_backbone', 'resnet50_ms_l1_backbone',
        #    'resnet101_ms_l123_backbone', 'resnet101_ms_l12_backbone', 'resnet101_ms_l1_backbone',
        #    'resnet18_dynamic_ms_l123', 'resnet18_dynamic_ms_l12', 'resnet18_dynamic_ms_l1',
        #    'resnet50_dynamic_ms_l123', 'resnet50_dynamic_ms_l12', 'resnet50_dynamic_ms_l1',
        #    'resnet101_dynamic_ms_l123', 'resnet101_dynamic_ms_l12', 'resnet101_dynamic_ms_l1',
        #    'resnet18_dynamic_ms_l123_backbone', 'resnet18_dynamic_ms_l12_backbone', 'resnet18_dynamic_ms_l1_backbone',
        #    'resnet50_dynamic_ms_l123_backbone', 'resnet50_dynamic_ms_l12_backbone', 'resnet50_dynamic_ms_l1_backbone',
        #    'resnet101_dynamic_ms_l123_backbone', 'resnet101_dynamic_ms_l12_backbone',
        #    'resnet101_dynamic_ms_l1_backbone']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnet18_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet18_dynamic-074db766.pth',
    'resnet50_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet50_dynamic-2c3b0201.pth',
    'resnet101_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet101_dynamic-c5f15780.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv3x3_dynamic(in_planes: int, out_planes: int,
                    stride: int = 1,
                    attention_in_channels: int = None) -> Conv2dDynamic:
    """3x3 convolution with padding"""
    return Conv2dDynamic(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=True, attention_in_channels=attention_in_channels)

def conv3x3_dynamic_lrd( in_planes: int, out_planes: int,
                    stride: int = 1, mode='mode0', squeeze=None,
                    attention_in_channels: int = None) -> Conv2d_LRD_Dynamic:
    return Conv2d_LRD_Dynamic( in_planes, out_planes, kernel_size=3, stride=stride, bias=True, padding=1,
                              mode=mode, squeeze=squeeze, attention_in_channels=attention_in_channels)
    

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def conv1x1_dynamic_lrd(in_planes: int, out_planes: int,
                    stride: int = 1, mode='mode0', squeeze=None,
                    attention_in_channels: int = None) -> Conv2d_LRD_Dynamic:
    return Conv2d_LRD_Dynamic(in_planes, out_planes, kernel_size=1, stride=stride, bias=True,
                              mode=mode, squeeze=squeeze, attention_in_channels=attention_in_channels)



class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
           
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockDynamic(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_dynamic(inplanes, planes, stride, attention_in_channels=inplanes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dynamic(planes, planes, attention_in_channels=inplanes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x, attention_x=x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlockDynamicLRD(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_dynamic(inplanes, planes, stride, attention_in_channels=inplanes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dynamic_lrd(planes, planes, attention_in_channels=inplanes)
        self.bn2 = norm_layer(int(planes))
        self.downsample = downsample
        self.stride = stride


    def forward(self,x):
    
        identity = x
        out = self.conv1(x, attention_x=x)
        out = self.bn1(out)
        out = self.relu(out)
        out, common, specific = self.conv2(out, attention_x=x)
        common = self.bn2(common)
        specific = self.bn2(specific)

        if self.downsample is not None:
            identity = self.downsample(x)

        common += identity
        specific += identity

        common = self.relu(common)
        specific = self.relu(specific)
        feature = torch.cat((common, specific), dim=1)

        return feature, common, specific

       


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
    
        self.base_width = base_width
        self.groups = groups
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (self.base_width / 64.)) * self.groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleneckDynamic(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckDynamic, self).__init__()
        if groups != 1:
            raise ValueError('BottleneckDynamic only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BottleneckDynamic")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_dynamic(width, width, stride, attention_in_channels=inplanes)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class BottleneckDynamicLRD(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        
        super(BasicBlockDynamicLRD, self).__init__()

    
        if groups != 1:
            raise ValueError('BottleneckDynamic only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BottleneckDynamic")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_dynamic_lrd(width, width, stride, attention_in_channels=inplanes)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self,x):
        # identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out, common, specific = self.conv2(out, attention_x=x)

        common = self.bn2(common)
        specific = self.bn2(specific)
        
        common = self.relu(common)
        specific = self.relu(specific)

        common = self.conv3(common)
        specific = self.conv3(specific)
        
        common = self.bn3(common)
        specific = self.bn3(specific)
        
        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            # c_identity = self.downsample(common)
            # s_identity = self.downsample(specific)

        common += identity
        specific += identity

        common = self.relu(common)
        specific = self.relu(specific)

        feature = torch.cat((common, specific), dim=1)

    
        return feature, common, specific



class ResNet(nn.Module):
    def __init__(
            self,
            blocks,
            first_layer,
            layers,
            num_class = None,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group


        ###################################################################
        self.num_class = num_class
        self.blocks = blocks
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        
        self.layer1 = self._make_layer(blocks[0], first_layer,  64, layers[0]) 
        self.layer2 = self._make_layer(blocks[1], first_layer, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(blocks[2], first_layer, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(blocks[3], first_layer, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_features = 512 * blocks[3].expansion
    
    def forward(self, x):
        if (BasicBlockDynamicLRD in self.blocks) or (BottleneckDynamicLRD in self.blocks):
            feature, commom, specific = self._forward_dynamic_lrd(x)
            return feature, commom, specific
        # elif (BasicBlockDynamic in self.blocks) or (BottleneckDynamic in self.blocks):
        #     self._forward_dynamic(x)
        else: 
            out = self._forward(x)
            return out

    def _make_layer(self, block_type, first_layer, planes, num_block,
                    stride:int=1, dilate: bool=False) -> nn.Sequential:
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block_type.expansion, stride),
                norm_layer(planes * block_type.expansion),
            )
  
        layers = []
        layers.append(first_layer(self.inplanes, planes, stride, downsample, self.groups,
                                 self.base_width, previous_dilation, norm_layer))

        

        for i in range(1, num_block+1):
            
            if (block_type == BasicBlockDynamicLRD) or (block_type == BottleneckDynamicLRD):
                if i == num_block:
                    self.inplanes = planes * block_type.expansion
                    layers.append(block_type(self.inplanes, planes))  
                else: 
                    if (block_type == BasicBlockDynamicLRD):
                        self.inplanes = planes * BasicBlockDynamic.expansion
                        layers.append(BasicBlockDynamic(self.inplanes, planes))
                        
                    elif (block_type == BottleneckDynamicLRD):
                        self.inplanes = planes * BottleneckDynamic.expansion
                        layers.append(BottleneckDynamic(self.inplanes, planes))
                
            else: 
                self.inplanes = planes * block_type.expansion
                layers.append( block_type(self.inplanes, planes))
        return nn.Sequential(*layers)

 
    def _forward_dynamic_lrd(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out, common, specific = self.layer4(x)

        common = self.avgpool(common)
        specific = self.avgpool(specific)
        
        common = torch.flatten(common, 1)
        specific = torch.flatten(specific, 1)
        features = torch.cat((common, specific), dim=1)

        # print('out:', features.shape)
        # print('common:', common.shape)
        # print('specific:', specific.shape)
        return features, common, specific


    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    



        
blocknames= [BasicBlock, Bottleneck, 
             BasicBlockDynamic, BottleneckDynamic, 
             BasicBlockDynamicLRD, BottleneckDynamicLRD]



def _resnet(
        blocks: List,
        first_layer,
        layers: List[int],
        pretrained: bool,
        progress: bool,
        model_arch: str = None,
        **kwargs: Any
    ) -> ResNet:
    
    model = ResNet(blocks, first_layer, layers, pretrained, **kwargs)

    if pretrained:
        if model_arch is None:
            raise ValueError('model_arch should be provided when pretrained is True.')
        state_dict = load_state_dict_from_url(model_urls[model_arch],
                                            progress=progress)
        
        ms_layer = load_state_dict(model, state_dict)

        for m in model.modules():
            if m in ms_layer:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        return model

        

@MODELS_REGISTRY.register()
def resnet18_dynamic(name, args) -> ResNet:

    model = _resnet([BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic], 
                    BasicBlock, [2, 2, 2, 2], 
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet18')
    args.__dict__[name]['out_features'] = model.out_features
    return model

@MODELS_REGISTRY.register()
def resnet18_dynamic_lrd(name, args) -> ResNet:

    model = _resnet([BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamicLRD], 
                    BasicBlock, [2, 2, 2, 2], 
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet18')
    args.__dict__[name]['out_features'] = model.out_features
    return model

@MODELS_REGISTRY.register()
def resnet34_dynamic(name, args) -> ResNet:

    model = _resnet([BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic], 
                    BasicBlock, [3, 4, 6, 3], 
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet34')
    args.__dict__[name]['out_features'] = model.out_features
    return model

@MODELS_REGISTRY.register()
def resnet34_dynamic_lrd(name, args) -> ResNet:

    model = _resnet([BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamic,BasicBlockDynamicLRD], 
                    BasicBlock, [3, 4, 6, 3], 
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet34')
    args.__dict__[name]['out_features'] = model.out_features
    return model

@MODELS_REGISTRY.register()
def resnet50_dynamic(name, args) -> ResNet:

    model = _resnet([BottleneckDynamic, BottleneckDynamic, BottleneckDynamic, BottleneckDynamic],
                    Bottleneck, [3, 4, 6, 3],
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet50',
                    )
    args.__dict__[name]['out_features'] = model.out_features
    return model

@MODELS_REGISTRY.register()
def resnet50_dynamic_lrd(name, args) -> ResNet:

    model = _resnet([BottleneckDynamic, BottleneckDynamic, BottleneckDynamic, BottleneckDynamicLRD],
                    Bottleneck, [3, 4, 6, 3],
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet50',
                    )
    args.__dict__[name]['out_features'] = model.out_features
    return model





@MODELS_REGISTRY.register()
def resnet18(name, args) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
        """
    model = _resnet([BasicBlock, BasicBlock, BasicBlock, BasicBlock],
                    BasicBlock, [2, 2, 2, 2],
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet18',
                    )
                    
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet34(name, args) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """

    model = _resnet([BasicBlock, BasicBlock, BasicBlock, BasicBlock],
                    BasicBlock, [3, 4, 6, 3],
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet34',
                    )
    
    
    
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50(name, args) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """

    model = _resnet([Bottleneck, Bottleneck,Bottleneck,Bottleneck],
                    Bottleneck,[3, 4, 6, 3],
                    pretrained=args.pretrained, progress=True,
                    model_arch='resnet50',
                    )
    
    args.__dict__[name]['out_features'] = model.out_features
    return model



#






