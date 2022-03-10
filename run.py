import torch
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime

from load_data import load_db
from nets import CCP
from runner import train, test

# TODO Build another net besides CCP, a second simple one
# TODO Get dataloading working with Fashion MNIST, CIFAR10
# TODO Build a function that looks at training & testing and builds a confusion matrix of results (maybe torch has a prebuild thing), also allows me to display samples that are in each bucket.


if __name__ == '__main__':
    DATASETS = ['MNIST', "FashionMNIST", "CIFAR10"]
    chosen_dataset = DATASETS[2]
    lr = 0.001
    epochs = 5
    batch_size = 64
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load Data
    train_loader, valid_loader, test_loader = load_db(chosen_dataset, transform, batch_size=batch_size)

    print(f"Dataset: {type(train_loader.dataset)}")
    sample_shape = train_loader.dataset[0][0].shape
    #check image is square since using only 1 side of it for the shape
    assert(sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(train_loader.dataset.classes)
    channels_in = sample_shape[0]

    # create the model.
    net = CCP(16, image_size=image_size, n_classes=n_classes)
    net.apply(net.weights_init)

    # # define the optimizer.
    opt = optim.SGD(net.parameters(), lr=lr)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    writer = SummaryWriter(f'runs/{chosen_dataset}/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        train(net, train_loader, opt, criterion, epoch, device, writer)
        test(net, valid_loader, criterion, epoch, device, writer)