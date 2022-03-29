import torch
import matplotlib.pyplot as plt
import collections

import nets
from load_data import load_dataset, load_testloader
from runner import test_to_analyze
from plots import plot_confusion_matrix
from utils import MODEL_PATH, get_activation


def compare_two_nets_by_sample(path1, path2, chosen_dataset):
    checkpoint1 = torch.load(path1)
    checkpoint2 = torch.load(path2)

    assert checkpoint1["apply_manipulation"] == checkpoint2["apply_manipulation"]
    assert str(checkpoint1["transform"]) == str(checkpoint2["transform"])
    assert checkpoint1["ind_to_keep"] == checkpoint2["ind_to_keep"]
    assert checkpoint1["binary_class"] == checkpoint2["binary_class"]

    y_pred_1, y_true_1, labels = test_net(checkpoint1, chosen_dataset)
    y_pred_2, y_true_2, _ = test_net(checkpoint2, chosen_dataset)

    assert all(val1 == val2 for (val1, val2) in zip(y_true_1, y_true_2))

    title = f"{chosen_dataset} x: {checkpoint1['n_degree']} y: {checkpoint2['n_degree']}"
    fig = plot_confusion_matrix(y_pred_1, y_pred_2, labels, title)
    plt.show()

    title = f"{chosen_dataset} x: {checkpoint1['n_degree']} y: {checkpoint2['n_degree']}"
    correct_1 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_1, y_true_1)]
    correct_2 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_2, y_true_2)]
    fig = plot_confusion_matrix(correct_1, correct_2, ["0", "1"], title)
    plt.show()


def test_net(checkpoint, chosen_dataset, register=[], activation=None):
    # Load Data in particular way
    train_dataset, test_dataset = load_dataset(
        chosen_dataset,
        checkpoint["transform"],
        apply_manipulation=checkpoint["apply_manipulation"],
        binary_class=checkpoint["binary_class"],
        ind_to_keep=checkpoint["ind_to_keep"]
    )
    # Initialize dataloaders
    test_loader = load_testloader(
        test_dataset,
        batch_size=checkpoint["batch_size"],
        num_workers=0,
        shuffle=True,
        seed=0,
    )

    print(f"Dataset: {type(test_loader.dataset)}")
    sample_shape = test_loader.dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(test_loader.dataset.classes)
    channels_in = sample_shape[0]

    net = getattr(nets, checkpoint.get("net_name", "CCP"))(
        checkpoint["hidden_size"],
        image_size=image_size,
        n_classes=n_classes,
        channels_in=channels_in,
        n_degree=checkpoint["n_degree"]
    )
    net.load_state_dict(checkpoint['model_state_dict'])

    # add hooks
    for lay in register:
        getattr(net, lay).register_forward_hook(get_activation(activation, lay))

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")

    y_pred, y_true = test_to_analyze(net, test_loader, device)
    labels = test_loader.dataset.class_to_idx.keys()

    return y_pred, y_true, labels


if __name__ == '__main__':
    paths = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{MODEL_PATH}/CIFAR10/**/sub*.ckpt", recursive=True)
    ))
    print(paths)

    compare_two_nets_by_sample(paths[0], paths[1], paths[0].split(os.sep)[-3], color='uniform')
