import torch
import torch.nn.functional as F
from collections import defaultdict
from temperature_scaling import ModelWithTemperature
import logging
from utils import load_model, load_checkpoint, load_yaml, set_seed, SCRATCH_PATH
from load_data import load_dataset, load_trainloader, load_testloader
import glob
import re
import sys

logger = logging.getLogger(__name__)

def load_checkpoint_config(paths, configs):
    assert len(paths) == len(configs)
    checkpoint_list = []
    config_list = []
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    for i in range(len(paths)):
        checkpoint_list.append(torch.load(paths[i], map_location=device))
        config_list.append(load_yaml(configs[i]))

    return checkpoint_list, config_list

def load_and_calibrate_models(checkpoint, config, device, valid_loader, temperature=False, save=True):
    # Load Model
    sample_shape = valid_loader.dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    num_classes = len(valid_loader.dataset.classes)
    channels_in = sample_shape[0]

    net = load_model(
        config['model'],
        image_size=image_size,  # this parameter doesn't matter for Resnet
        num_classes=num_classes,
        channels_in=channels_in
    )
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    if temperature:
        # Put into Temperature class
        net = ModelWithTemperature(net, device)
        # Calibrate temperature
        net.set_temperature(valid_loader)
        logging.info(f"Temperature: {net.temperature}")

    if save:
        pass
        # Save checkpoint with temperature
        # Should I overwrite old one?
        # would need location to save...

    return net

def test_ensemble(temp_models, test_loader, device):
    # TODO all tensors not on the same device...
    correct, total = 0, 0
    model_pred = defaultdict(list)
    model_softmax = defaultdict(list)
    y_true = []
    ensemble_hat = []
    correct_list = [0]*(len(temp_models))
    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        img = data[0].to(device)
        label = data[1].to(device)
        y_true.extend(label.tolist())
        for i, net in enumerate(temp_models):
            net.eval()
            with torch.no_grad():
                pred = net(img)
                # initialize ensemble pred with zeros on first iter and add to it
            if i == 0:
                ensemble_pred = torch.zeros(pred.shape, device=device)
            ensemble_pred += pred

            pred_softmax = F.softmax(pred, dim=1)
            _, predicted = pred.max(1)
            # TODO do we need .item()
            correct_softmax = pred_softmax[[i for i in range(0, len(label))], label.tolist()]
            #TODO check tolist is necessary
            model_pred[f'model_{i}'].extend(predicted.tolist())f
            model_softmax[f'model_{i}_softmax'].extend(correct_softmax.tolist())
            correct_list[i] += predicted.eq(label).sum().item()

        #take average of ensemble predictions
        ensemble_pred = 1/(len(temp_models)) * ensemble_pred
        _, ensemble_predicted = ensemble_pred.max(1)
        ensemble_hat.extend(ensemble_predicted.tolist())
        total += label.size(0)
        correct += ensemble_predicted.eq(label).sum().item()

    ens_acc = float(correct) / total
    correct_list = [x / total for x in correct_list]
    correct_list.append(ens_acc)

    # list of accuracies, true labels, ensemble_predictions, model_predictions, model_softmax
    return correct_list, y_true, ensemble_hat, model_pred, model_softmax


if __name__ == "__main__":
    # These need to be written to work with GPU's
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    paths = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{SCRATCH_PATH}/logs/CIFAR100/NCPS*/latest*", recursive=True)
    ))
    print(paths)

    configs = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{SCRATCH_PATH}/logs/CIFAR100/NCPS*/r_ncp*", recursive=True)
    ))
    print(configs)
    configs = configs * 4
    # load each model into memory
    checkpoint_list, config_list = load_checkpoint_config(paths, configs)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    logging.info(f"Running on: {device}")

    # TODO what to do if they don't fit in memory???
    temp_models = []
    for checkpoint, config in zip(checkpoint_list, config_list):
        model = load_and_calibrate_models(checkpoint, config, device, save=False)
        temp_models.append(model)

    config = config_list[0]
    train_dataset, test_dataset = load_dataset(
        **config['dataset']
    )
    # Initialize dataloaders
    test_loader = load_testloader(
        test_dataset,
        batch_size=config['dataloader']['batch_size'],
        num_workers=0,
        shuffle=False,
    )
    correct_list, y_true, ensemble_hat, model_pred, model_softmax = test_ensemble(temp_models, test_loader, device)

