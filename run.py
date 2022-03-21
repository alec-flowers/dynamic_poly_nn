import torch
import os
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import MODEL_PATH, path_exist, count_parameters

from load_data import load_db
from nets import CCP, NCP
from runner import train, test, load_checkpoint, train_profile

# TODO Look into coding T-SNE
# TODO Think of and run experiments
# TODO Look into FFCV api for dataloading FAST - https://docs.ffcv.io/basics.html


if __name__ == '__main__':
    # Parameters
    save = False
    DATASETS = ['MNIST', "FashionMNIST", "CIFAR10"]
    chosen_dataset = DATASETS[1]
    checkpoint = None # str(MODEL_PATH) + f"/{chosen_dataset}/20220311-170934.ckpt"
    confusion_matrix = False
    subset = True
    num_workers = 4
    apply_manipulation = OneClassDataset
    ind_to_keep = None
    binary_class = 6


    # Hyperparameters
    lr = 0.001
    epochs = 5
    batch_size = 64
    n_degree = 3
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load Data in particular way
    train_dataset, test_dataset = load_dataset(
        chosen_dataset,
        transform,
        apply_manipulation=apply_manipulation,
        binary_class=binary_class
    )
    # Initialize dataloaders
    train_loader, valid_loader, test_loader = split_and_load_dataloader(
        train_dataset,
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        valid_ratio=0.2,
        seed=0,
        subset=subset
    )


    print(f"Dataset: {type(train_loader.dataset)}")
    sample_shape = train_loader.dataset[0][0].shape
    #check image is square since using only 1 side of it for the shape
    assert(sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(train_loader.dataset.classes)
    channels_in = sample_shape[0]

    # create the model.
    net = CCP(16, image_size=image_size, n_classes=n_classes, channels_in=channels_in, n_degree=n_degree)
    # net = NCP(16, 8, image_size=image_size, n_classes=n_classes, channels_in=channels_in, n_degree=n_degree, skip=True)
    net.apply(net.weights_init)
    print(f"Degree: {n_degree} Num Parameters: {count_parameters(net)}")

    # `define the optimizer.
    opt = optim.SGD(net.parameters(), lr=lr)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    if checkpoint:
        load_checkpoint(net, opt, checkpoint)

    writer = SummaryWriter(f'runs/{chosen_dataset}/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        train_profile(net, train_loader, opt, criterion, epoch, device, writer, confusion_matrix)
        _, _ = test(net, valid_loader, criterion, epoch, device, writer)
    if save:
        path = str(MODEL_PATH) + f"/{chosen_dataset}"
        path_exist(path)
        torch.save(
            {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'chosen_dataset': chosen_dataset,
                'subset': subset,
                "num_workers": num_workers,
                "apply_manipulation": apply_manipulation,
                "ind_to_keep": ind_to_keep,
                "binary_class": binary_class,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "n_degree": n_degree,
                "transform": transform,
             }, path + "/" + CUSTOM_SAVE + "_" + path_time + ".ckpt")