'''Model for Π-net based 2nd degree blocks without activation functions:
https://ieeexplore.ieee.org/document/9353253 (or https://arxiv.org/abs/2006.13026).
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_activ=True, use_alpha=False, n_lconvs=0,
                 norm_local=None, kern_loc=1, norm_layer=None, norm_x=-1,
                 n_xconvs=0, use_only_first_conv=False, **kwargs):
        super(BasicBlock, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs
        self.n_xconvs = n_xconvs
        self.use_only_first_conv = use_only_first_conv

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = self._norm_layer(planes)
        if not self.use_only_first_conv:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = self._norm_layer(planes)

        self.shortcut = nn.Sequential()
        planes1 = in_planes
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(self.expansion*planes)
            )
            planes1 = self.expansion * planes
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.monitor_alpha = []
        self.normx = self._norm_x(planes1)
        # # define 'local' convs, i.e. applied only to second order term, either
        # # on x or after the multiplication.
        self.def_local_convs(planes, n_lconvs, kern_loc, self._norm_local, key='l')
        self.def_local_convs(planes1, n_xconvs, kern_loc, self._norm_x, key='x')

    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        if not self.use_only_first_conv:
            out = self.bn2(self.conv2(out))
        out1 = out + self.shortcut(x)
        # # normalize the x (shortcut one).
        x1 = self.normx(self.shortcut(x))
        x1 = self.apply_local_convs(x1, self.n_xconvs, key='x')
        out_so = out * x1
        out_so = self.apply_local_convs(out_so, self.n_lconvs, key='l')
        if self.use_alpha:
            out1 += self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out_so
        out = self.activ(out1)
        return out

    def def_local_convs(self, planes, n_lconvs, kern_loc, func_norm, key='l', typet='conv'):
        """ Aux function to define the local conv/fc layers. """
        if n_lconvs > 0:
            s = '' if n_lconvs == 1 else 's'
            # print('Define {} local {}{}{}.'.format(n_lconvs, key, typet, s))
            if typet == 'conv':
                convl = partial(nn.Conv2d, in_channels=planes, out_channels=planes,
                                kernel_size=kern_loc, stride=1, padding=0, bias=False)
            else:
                convl = partial(nn.Linear, planes, planes)
            for i in range(n_lconvs):
                setattr(self, '{}{}{}'.format(key, typet, i), convl())
                setattr(self, '{}bn{}'.format(key, i), func_norm(planes))

    def apply_local_convs(self, out_so, n_lconvs, key='l'):
        if n_lconvs > 0:
            for i in range(n_lconvs):
                out_so = getattr(self, '{}conv{}'.format(key, i))(out_so)
                out_so = getattr(self, '{}bn{}'.format(key, i))(out_so)
        return out_so


class ModelNCPS(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None,
                 pool_adapt=False, n_channels=[64, 128, 256, 512], ch_in=3, **kwargs):
        super(ModelNCPS, self).__init__()
        self.in_planes = n_channels[0]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)

        self.conv1 = nn.Conv2d(ch_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(n_channels[-1] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                norm_layer=self._norm_layer, **kwargs))
            self.in_planes = planes * block.expansion
        # # cheeky way to get the activation from the layer1, e.g. in no activation case.
        self.activ = layers[0].activ
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def NCPS(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return ModelNCPS(BasicBlock, num_blocks, **kwargs)
