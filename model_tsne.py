import torch
import os
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import MODEL_PATH, path_exist, count_parameters, load_checkpoint
from sklearn.manifold import TSNE
import numpy as np

from load_data import load_dataset, only_use_certain_class, OneClassDataset, split_and_load_dataloader
from nets import CCP, NCP
from runner import train, test, train_profile, test_to_analyze
from plots import plot_per_class_accuracy, plot_tsne, plot_confusion_matrix
import collections
import matplotlib.pyplot as plt


if __name__ == '__main__':
    DATASETS = ['MNIST', "FashionMNIST", "CIFAR10"]
    chosen_dataset = DATASETS[1]
    n_degree = 16
    file = "20220321-165247.ckpt"

    # which layers to register hooks
    REGISTER = ["Id_U4", "Id_U8", "Id_U12", f"Id_U{n_degree}"]

    CHECK_PATH = str(MODEL_PATH) + f"/{chosen_dataset}/{n_degree}/{file}"

    checkpoint = torch.load(CHECK_PATH)

    # Load Data in particular way
    train_dataset, test_dataset = load_dataset(
        chosen_dataset,
        checkpoint["transform"],
        apply_manipulation=checkpoint["apply_manipulation"],
        binary_class=checkpoint["binary_class"],  #TODO this will likely throw error
        ind_to_keep=checkpoint["ind_to_keep"]
    )
    # Initialize dataloaders
    train_loader, valid_loader, test_loader = split_and_load_dataloader(
        train_dataset,
        test_dataset,
        batch_size=checkpoint["batch_size"],
        num_workers=checkpoint["num_workers"],
        shuffle=True,
        valid_ratio=0.2,
        seed=0,
        subset=checkpoint["subset"]
    )

    print(f"Dataset: {type(train_loader.dataset)}")
    sample_shape = train_loader.dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(train_loader.dataset.classes)
    channels_in = sample_shape[0]

    # create hook
    activation = collections.defaultdict(list)
    def get_activation(name):
        def hook(inst, inp, out):
            activation[name].append(out.detach().numpy())
        return hook

    # load model
    net = CCP(checkpoint["hidden_size"], image_size=image_size, n_classes=n_classes, channels_in=channels_in, n_degree=n_degree)
    net.load_state_dict(checkpoint['model_state_dict'])

    # add hooks
    for lay in REGISTER:
        getattr(net, lay).register_forward_hook(get_activation(lay))

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")
    criterion = torch.nn.CrossEntropyLoss().to(device)

    labels = test_loader.dataset.class_to_idx.keys()
    y_pred, y_true = test_to_analyze(net, test_loader, device)

    plot_per_class_accuracy(y_true, y_pred, labels)
    fig = plot_confusion_matrix(y_true, y_pred, labels)
    plt.show()

    tsne = TSNE(n_components=2, verbose=1, perplexity=20, learning_rate='auto')
    for lay in REGISTER:
        title = [chosen_dataset, n_degree, lay]
        tsne_results = tsne.fit_transform(np.concatenate(activation[lay]))
        plot_tsne(
            tsne_results[:, 0],
            tsne_results[:, 1],
            y_true,
            n_colors=len(test_loader.dataset.class_to_idx),
            title=title
        )
