import torch
from torch import nn
import os
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import MODEL_PATH, path_exist, count_parameters, load_checkpoint, WRITER_PATH
import time

from load_data import load_dataset, load_trainloader, SpecialistDataset
from networks import nets
from runner import train, test

if __name__ == '__main__':

    custom_save_list = ["spec_tl_01", "spec_tl_234", "spec_tl_567", "spec_tl_89"]
    save = True
    DATASETS = ['MNIST', "FashionMNIST", "CIFAR10"]
    chosen_dataset = DATASETS[0]
    checkpoint_list = [f"{MODEL_PATH}/{chosen_dataset}/2/_e50_20220403-165729.ckpt",
                  f"{MODEL_PATH}/{chosen_dataset}/4/_e50_20220403-165927.ckpt",
                  f"{MODEL_PATH}/{chosen_dataset}/8/_e50_20220403-170113.ckpt",
                  f"{MODEL_PATH}/{chosen_dataset}/16/_e50_20220403-170305.ckpt",]
    for path in checkpoint_list:
        assert os.path.exists(path)

    confusion_matrix = False
    subset = False
    num_workers = 4
    apply_manipulation = SpecialistDataset
    ind_to_keep_list = [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9]]
    binary_class = None
    g = torch.Generator()
    g.manual_seed(0)

    # Hyperparameters
    net_name = "CCP"
    lr_list = [0.001, 0.001, 0.0003, .00006]
    epochs_list = [20, 20, 20, 20]
    batch_size = 64
    n_degree_list = [2, 4, 8, 16]
    hidden_size = 16  # p = 2n + d
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    n_classes_for_loading = 10


# ============================== RUN ============================== #
    for i in range(len(epochs_list)):
        epochs = epochs_list[i]
        n_degree = n_degree_list[i]
        lr = lr_list[i]
        ind_to_keep = ind_to_keep_list[i]
        CUSTOM_SAVE = custom_save_list[i]
        checkpoint = checkpoint_list[i]

        path_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folders = f"{chosen_dataset}/{n_degree}"
        filename = f"{CUSTOM_SAVE}_e{epochs}_{path_time}"
        path_exist(f"{WRITER_PATH}/{folders}", f"{MODEL_PATH}/{folders}")
        W_PATH = f"{WRITER_PATH}/{folders}/{filename}"
        M_PATH = f"{MODEL_PATH}/{folders}/{filename}.ckpt"

        # Load Data in particular way
        train_dataset, test_dataset = load_dataset(
            chosen_dataset,
            transform,
            apply_manipulation=apply_manipulation,
            binary_class=binary_class,
            ind_to_keep=ind_to_keep,
            color=None,
            generator=g
        )
        # Initialize dataloaders
        train_loader, valid_loader = load_trainloader(
            train_dataset,
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
        net = getattr(nets, net_name)(
            hidden_size=hidden_size,
            image_size=image_size,
            n_classes=n_classes_for_loading,
            channels_in=channels_in,
            n_degree=n_degree
        )
        net.apply(net.weights_init)
        num_params = count_parameters(net)
        print(f"Degree: {n_degree} Num Parameters: {count_parameters(net)}")

        # `define the optimizer.
        opt = optim.SGD(net.parameters(), lr=lr)
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')
        print(f"Running on: {device}")

        if checkpoint:
            load_checkpoint(net, checkpoint, opt)

        #override last layer with my own
        net.C = nn.Linear(hidden_size, n_classes, bias=True)

        if device == 'cuda':
            print(f"Num Devices: {torch.cuda.device_count()}")
            dev = torch.cuda.current_device()
            print(f"Device Name: {torch.cuda.get_device_name(dev)}")
        criterion = torch.nn.CrossEntropyLoss().to(device)

        writer = SummaryWriter(W_PATH)

        start_time = time.time()
        for epoch in range(epochs):
            train(net, train_loader, opt, criterion, epoch, device, writer, confusion_matrix)
            if epoch % 5 == 0:
                test(net, valid_loader, criterion, epoch, device, writer)
        writer.flush()
        total_time = time.time() - start_time
        print(f"Total Training Time: {total_time:.1f} seconds")

        if save:
            print("Saving Model")
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
                    "hidden_size": hidden_size,
                    "total_time": total_time,
                    "net_name": net_name,
                    "num_params": num_params
                 }, M_PATH)