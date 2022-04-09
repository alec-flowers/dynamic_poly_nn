import argparse
import importlib
import random
import torch
import os
from torch import optim
import yaml
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import path_exist, count_parameters, load_checkpoint, REPO_ROOT, CONFIGS_PATH, save_checkpoint, load_model, set_seed, init_weights
import time

from load_data import load_dataset, load_trainloader
from networks import nets
from runner import train, test

# TODO - Learning Rate Milestones

def parse_args():
    parser = argparse.ArgumentParser(description="run networks")
    parser.add_argument("--config_name", type=str,
                        required=True, default='yml',
                        help="Name of YAML config file.")

    return parser.parse_args()

def run(args):
    curdir = os.path.abspath(os.path.curdir)
    print(f"Current Running Directory: {curdir}")
    print(f"Repo Root: {REPO_ROOT}")

    try:
        with open(os.path.join(CONFIGS_PATH, args.config_name), 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    if config['seed'] is None:
        seed = random.randint(1, 1000)
    else:
        seed = config['seed']
    set_seed(seed)

    # Create all paths we will need
    custom_save = config["custom_save"]
    path_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folders_for_save = f"logs/{config['dataset']['name']}/{config['model']['name']}_{custom_save}_{path_time}"
    savepath = f"{curdir}/{folders_for_save}"
    path_exist(savepath)

    with open(os.path.join(savepath, f"r_{args.config_name}"), 'w') as file:
        yaml.dump(
            config,
            file,
            default_flow_style=False,
        )

    # Load Data in particular way
    train_dataset, test_dataset = load_dataset(**config["dataset"])

    # Initialize dataloaders
    train_loader, valid_loader = load_trainloader(
        train_dataset,
        seed=seed,
        **config["dataloader"]
    )

    #TODO may need to change this with RESNET
    print(f"Length of iters per epoch: {len(train_loader)}. Length of valid batches: {len(valid_loader)}. Size of img: {train_loader.dataset[0][0].shape}")
    sample_shape = train_loader.dataset[0][0].shape
    assert(sample_shape[1] == sample_shape[2]), "Image is not square, but only use one side"
    image_size = sample_shape[1]
    num_classes = len(train_loader.dataset.classes)
    channels_in = sample_shape[0]

    # create the model.
    net = load_model(
        config['model'],
        image_size=image_size, # this parameter doesn't matter for Resnet
        num_classes=num_classes,
        channels_in=channels_in)

    net.init = config['model']['args'].get('init', None)
    init_weights(net)

    num_params = count_parameters(net)
    deg = config['model']['args'].get('n_degree', config['model']['args'].get('num_blocks', None))
    print(f"Degree: {deg} Num Parameters: {num_params}")

    # `define the optimizer.
    decay = config['training_info'].get('weight_dec', 5e-4)
    opt = optim.SGD(net.parameters(), lr=config['training_info']['lr'],
                    momentum=0.9, weight_decay=decay)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    start_epoch = -1
    if config['checkpoint']:
        checkpoint = load_checkpoint(net, config['checkpoint'], opt)
        start_epoch = checkpoint['epoch']

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.to(device)

    mil = config['training_info'].get('lr_milestones', [40, 60, 80, 100])
    gamma = config['training_info'].get('lr_gamma', 0.1)
    # TODO May be something finicky here when loading a model and training.
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=mil,
                                               gamma=gamma, last_epoch=start_epoch)
    writer = SummaryWriter(savepath)

    start_time = time.time()
    old_acc = 0
    for epoch in range(start_epoch+1, config['training_info']['epochs']+1):
        train(net, train_loader, opt, criterion, epoch, device, writer, display_interval=config['training_info']['display_interval'])

        if epoch % config['training_info']['save_every_epoch'] == 0 and epoch > 0:
            elapsed_time = time.time()-start_time
            save_checkpoint(net, opt, epoch, elapsed_time, savepath, f"latest_e{epoch}_t{elapsed_time:.0f}")

        if epoch % config['training_info']['test_every_epoch'] == 0:
            acc = test(net, valid_loader, criterion, epoch, device, writer)
            if acc > old_acc:
                old_acc = acc
                elapsed_time = time.time() - start_time
                save_checkpoint(net, opt, epoch, elapsed_time, savepath, f"best_model")

        scheduler.step()

    writer.flush()
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.1f} seconds")


if __name__ == '__main__':
    args = parse_args()
    run(args)
