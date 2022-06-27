import torch
from torch import nn


class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.sample = None

    def batch(self, size):
        binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
        self.sample = binomial.sample((1, size[-1])) * (1.0/(1-self.p))

    def forward(self, X):
        if self.training:
            return X * self.sample
        return X


class CCP_4(nn.Module):
    def __init__(self, hidden_size, image_size=28, channels_in=1, n_degree=4, bias=False, num_classes=10):
        super(CCP_4, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_degree = n_degree
        hidden = [64, 64, 16, 16]
        drop = [.5, .5, .1, .1]
        for i in range(1, self.n_degree + 1):
            setattr(self, f'U{i}', nn.Linear(self.total_image_size, hidden[i-1], bias=bias))
            setattr(self, f'Id_U{i}', nn.Identity())
            setattr(self, f'dropout{i}', nn.Dropout(p=drop[i-1]))
            setattr(self, f"maxpool{i}", nn.AdaptiveMaxPool1d(hidden[i-1]))
        self.C = nn.Linear(self.hidden_size, self.num_classes, bias=True)
        # self.norm = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.U1(h)
        #self.dropout.batch(out.size())
        #out = self.dropout(self.norm(out))
        out = self.dropout1(out)
        for i in range(2, self.n_degree + 1):
            #inner = self.dropout(self.norm(getattr(self, f'U{i}')(h) * out))
            inner = getattr(self, f'U{i}')(h) * getattr(self, f'maxpool{i}')(out)
            # inner = getattr(self, f"dropout{i}")(inner)
            out = getattr(self, f"Id_U{i}")(inner + getattr(self, f'maxpool{i}')(out))
        out = self.C(out)
        return out


class CCP(nn.Module):
    def __init__(self, hidden_size, image_size=28, channels_in=1, n_degree=4, bias=False, num_classes=10):
        super(CCP, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_degree = n_degree
        self.dropout1 = nn.Dropout(p=.5)
        for i in range(1, self.n_degree + 1):
            setattr(self, f'U{i}', nn.Linear(self.total_image_size, self.hidden_size, bias=bias))
            setattr(self, f'Id_U{i}', nn.Identity())
            setattr(self, f'dropout{i}', nn.Dropout(p=.2))
        self.C = nn.Linear(self.hidden_size, self.num_classes, bias=True)
        # self.norm = nn.BatchNorm1d(num_features=hidden_size)

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.U1(h)
        #self.dropout.batch(out.size())
        #out = self.dropout(self.norm(out))
        out = self.dropout1(out)
        for i in range(2, self.n_degree + 1):
            #inner = self.dropout(self.norm(getattr(self, f'U{i}')(h) * out))
            inner = getattr(self, f'U{i}')(h) * out
            # inner = getattr(self, f"dropout{i}")(inner)
            out = getattr(self, f"Id_U{i}")(inner + out)
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


class CCP_non_poly(nn.Module):
    def __init__(self, hidden_size, image_size=28, channels_in=1, n_degree=4, bias=False, num_classes=10):
        super(CCP_non_poly, self).__init__()
        self.image_size = image_size
        self.channels_in = channels_in
        self.total_image_size = self.image_size * self.image_size * channels_in
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_degree = n_degree
        self.activ = nn.ReLU(inplace=True)
        for i in range(1, self.n_degree + 1):
            setattr(self, f'U{i}', nn.Linear(self.total_image_size, self.hidden_size, bias=bias))
            setattr(self, f'Id_U{i}', nn.Identity())
        self.C = nn.Linear(self.hidden_size, self.num_classes, bias=True)
        # TODO this is dumb as cross entropy applies softmax... this does it twice...
        self.final_activ = nn.Softmax()

    def forward(self, z):
        h = z.view(-1, self.total_image_size)
        out = self.activ(self.U1(h))
        for i in range(2, self.n_degree + 1):
            out = self.activ(getattr(self, f"Id_U{i}")(getattr(self, f'U{i}')(h) + out))
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