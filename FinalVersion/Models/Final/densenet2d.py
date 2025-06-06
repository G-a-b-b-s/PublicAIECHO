import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 norm_type='Unknown'):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', get_norm(norm_type, num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', get_norm(norm_type, bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  # noqa
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_type='Unknown'):  # noqa
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,
                                norm_type=norm_type)  # noqa
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 norm_type='Unknown'):
        super(_Transition, self).__init__()
        self.add_module('norm', get_norm(norm_type, num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,  # noqa
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet2d(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer  # noqa
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, input_channels = 30, growth_rate=32, block_config=(6, 12, 24, 16), 
                 norm_type='Unknown', num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):  # noqa

        super(DenseNet2d, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # noqa
            ('norm0', get_norm(norm_type, num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, norm_type=norm_type,  # noqa
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)  # noqa
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    norm_type=norm_type)  # noqa
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', get_norm(norm_type, num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        ################################
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        ###############################
        return x

def densenet_small(input_channels = 30, num_init_features=64, growth_rate=32, block_config=(6, 12, 24),  # noqa
                     norm_type='BatchNorm', num_classes=4,**kwargs):
    """Densenet-121 model from"""
    model = DenseNet2d(input_channels = input_channels, num_init_features=num_init_features, growth_rate=growth_rate, block_config=block_config,  
                     norm_type=norm_type, num_classes=4, **kwargs)
    return model


def densenet121(input_channels = 30, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),  # noqa
                     norm_type='BatchNorm', num_classes=4, **kwargs):
    """Densenet-121 model from"""
    model = DenseNet2d(input_channels = input_channels, num_init_features=num_init_features, growth_rate=growth_rate, block_config=block_config,  
                     norm_type=norm_type, num_classes=4, **kwargs)
    return model


def densenet169(input_channels = 30, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),  # noqa
                     norm_type='BatchNorm', num_classes=4, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    """
    model = DenseNet2d(input_channels = input_channels, num_init_features=num_init_features, growth_rate=growth_rate, block_config=block_config,  
                     norm_type=norm_type, num_classes=num_classes, **kwargs)
    return model


def densenet201( input_channels = 30, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),  # noqa
                     norm_type='BatchNorm',num_classes=4,**kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    """
    model = DenseNet2d(input_channels = input_channels, num_init_features=num_init_features, growth_rate=growth_rate, block_config=block_config,  
                     norm_type=norm_type, num_classes=num_classes, **kwargs)
    return model


def densenet161(input_channels = 30, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),  
                     norm_type='BatchNorm',num_classes=4, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ 
    """
    model = DenseNet2d(input_channels = input_channels, num_init_features=num_init_features, growth_rate=growth_rate, block_config=block_config,  
                     norm_type=norm_type, num_classes=num_classes, **kwargs)
    return model
