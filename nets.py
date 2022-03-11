import torch
from torch import nn


class CCP(nn.Module):
    def __init__(self, hidden_size, image_size=28, channels_in=1, n_degree=4, bias=False, n_classes=10):
        super(CCP, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, 'U{}'.format(i), nn.Linear(self.total_image_size, self.hidden_size, bias=bias))
        self.C = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.U1(h)
        for i in range(2, self.n_degree + 1):
            out = getattr(self, 'U{}'.format(i))(h) * out + out
        out = self.C(out)
        return out

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'CCP_Conv' and classname != 'NCP_Conv':
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        # print('initializing {}'.format(classname))


class NCP(nn.Module):
    def __init__(self, hidden_size, b_size, image_size=28, channels_in=1, n_degree=4, bias=False, n_classes=10, skip=False):
        super(NCP, self).__init__()
        self.skip = skip
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.b_size = b_size
        self.n_classes = n_classes
        self.n_degree = n_degree
        for i in range(1, self.n_degree + 1):
            setattr(self, f'A{i}', nn.Linear(self.total_image_size, self.hidden_size, bias=bias))
            setattr(self, f'B{i}', nn.Linear(self.b_size, self.hidden_size, bias=bias))
            setattr(self, f'b{i}', nn.Parameter(torch.normal(torch.zeros(self.b_size), torch.ones(self.b_size))))
            if i > 1:
                setattr(self, f'S{i}', nn.Linear(self.hidden_size, self.hidden_size, bias=bias))

        self.C = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.A1(h) * self.B1(self.b1)
        for i in range(2, self.n_degree + 1):
            tmp = getattr(self, f'A{i}')(h) * (getattr(self, f'S{i}')(out) + getattr(self, f'B{i}')(getattr(self, f'b{i}')))
            if self.skip:
                out = tmp + out
            else:
                out = tmp
        out = self.C(out)
        return out

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname != 'CCP_Conv' and classname != 'NCP_Conv':
            m.weight.data.normal_(0.0, 0.01)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        # print('initializing {}'.format(classname))