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