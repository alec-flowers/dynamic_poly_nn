'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
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
                 norm_local=None, kern_loc=1, norm_layer=None, use_lactiv=False, norm_x=-1,
                 n_xconvs=0, what_lactiv=-1, use_only_first_conv=False, div_factor=1, **kwargs):
        super(BasicBlock, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs
        self.n_xconvs = n_xconvs
        self.use_lactiv = self.use_activ and use_lactiv
        self.use_only_first_conv = use_only_first_conv
        self.div_factor = div_factor

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
        if self.use_lactiv:
            if what_lactiv == -1:
                ac1 = nn.ReLU(inplace=True)
            elif what_lactiv == 1:
                ac1 = nn.Softmax2d()
            elif what_lactiv == 2:
                ac1 = nn.LeakyReLU(inplace=True)
        self.lactiv = partial(ac1) if self.use_lactiv else lambda x: x
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
        if self.div_factor != 1:
            out_so = out_so * 1. / self.div_factor
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
                convl = nn.Conv2d(in_channels=planes, out_channels=planes,
                                kernel_size=kern_loc, stride=1, padding=0, bias=False)
            else:
                convl = partial(nn.Linear, planes, planes)
            for i in range(n_lconvs):
                setattr(self, '{}{}{}'.format(key, typet, i), convl)
                setattr(self, '{}bn{}'.format(key, i), func_norm(planes))

    def apply_local_convs(self, out_so, n_lconvs, key='l'):
        if n_lconvs > 0:
            for i in range(n_lconvs):
                out_so = getattr(self, '{}conv{}'.format(key, i))(out_so)
                out_so = self.lactiv(getattr(self, '{}bn{}'.format(key, i))(out_so))
        return out_so


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, use_activ=True, use_alpha=False, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_activ = use_activ

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = self._norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self._norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = self._norm_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(self.expansion*planes)
            )
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.monitor_alpha = []

    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.activ(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out1 = out + self.shortcut(x)
        if self.use_alpha:
            out1 += self.alpha * out * self.shortcut(x)
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out * self.shortcut(x)
        out = self.activ(out1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None, out_activ=False,
                 pool_adapt=False, n_channels=[64, 128, 256, 512], channels_in=3, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = n_channels[0]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.out_activ = out_activ
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)

        self.conv1 = nn.Conv2d(channels_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(n_channels[-1] * block.expansion, num_classes)
        # # if linear case and requested, include an output non-linearity.
        cond = self.out_activ and self.activ(-100) == -100
        self.oactiv = partial(nn.ReLU(inplace=True)) if cond else lambda x: x
        # print('output non-linearity: #', self.out_activ, cond)


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
        if self.out_activ:
            out = self.oactiv(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNetnew(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return ResNet(BasicBlock, num_blocks, **kwargs)

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)


if __name__ == '__main__':
    def test():
        net = ResNetnew(n_lconvs=2, n_xconvs=2)
        y = net(torch.randn(1,3,32,32))
        print(y.size())

    test()
