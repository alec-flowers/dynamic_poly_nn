import sys
import torch
import time
import logging
from torch.profiler import tensorboard_trace_handler
from plots import get_confusion_matrix, plot_confusion_matrix, plot_one
import numpy as np

logger = logging.getLogger(__name__)

def train(net, train_loader, optimizer, criterion, epoch, device, writer, scheduler=None, confusion_matrix=False, profiler=None, display_interval=100):
    """ Perform single epoch of the training."""
    net.train()
    running_loss, correct, total, train_loss, acc = 0, 0, 0, 0, 0
    start_time = time.time()
    iters = len(train_loader)
    for idx, data_dict in enumerate(train_loader):
        img = data_dict[0]
        label = data_dict[1]
        inputs, label = img.to(device), label.to(device)
        # efficiency increase
        optimizer.zero_grad(set_to_none=True)
        pred = net(inputs)
        loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(epoch + idx / iters)
        if profiler:
            profiler.step()

        total += label.size(0)
        running_loss += loss.item()

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(label).sum()

        train_loss = running_loss/total
        acc = float(correct)/total
        if idx % display_interval == 0 and idx > 0:
            m2 = ('Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            logger.info(m2.format(epoch, idx, len(train_loader), float(train_loss), acc))
    total_time = time.time() - start_time
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", acc, epoch)
    writer.add_scalar("1 Epoch Time/Train", total_time, epoch)
    writer.add_scalar("lr/Train", scheduler.get_last_lr()[0], epoch)
    # writer.flush()
    if confusion_matrix:
        writer.add_figure("Confusion/Train", get_confusion_matrix(train_loader, net, device), epoch)
        # writer.flush()


def train_profile(net, train_loader, optimizer, criterion, epoch, device, writer, confusion_matrix=False, display_interval=100):
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=6,
                repeat=1),
            on_trace_ready=tensorboard_trace_handler(writer.log_dir),
            with_stack=True
    ) as profiler:
        train(net=net,
              train_loader=train_loader,
              optimizer=optimizer,
              criterion=criterion,
              epoch=epoch,
              device=device,
              writer=writer,
              confusion_matrix=confusion_matrix,
              profiler=profiler,
              display_interval=display_interval)


def test(net, test_loader, criterion, epoch, device, writer):
    """ Perform testing, i.e. run net on test_loader .data
        and return the accuracy. """
    net.eval()
    predicted_list = []
    label_list = []
    correct, total, running_loss = 0, 0, 0
    start_time = time.time()
    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        img = data[0].to(device)
        label = data[1].to(device)
        with torch.no_grad():
            pred = net(img)
            loss = criterion(pred, label)

        total += label.size(0)
        running_loss += loss.item()

        _, predicted = pred.max(1)
        # TODO save at items not tensors
        # predicted_list.extend(predicted.numpy().astype(np.int16))
        correct += predicted.eq(label).sum().item()
        # label_list.extend(label.numpy().astype(np.int16))
    total_time = time.time() - start_time
    test_loss = running_loss / total
    acc = float(correct) / total

    writer.add_scalar("Loss/Test", test_loss, epoch)
    writer.add_scalar("Accuracy/Test", acc, epoch)
    writer.add_scalar("Accuracy/Test", acc, epoch)
    writer.add_scalar("1 Epoch Time/Test", total_time, epoch)


    logger.info(f'Epoch {epoch} (Validation - Loss: {test_loss:.03f} & Accuracy: {acc:.03f}')
    return acc, predicted_list, label_list


def test_to_analyze(net, loader, device):
    y_pred = []
    y_true = []
    net.eval()
    for (idx, data) in enumerate(loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(loader)))
        sys.stdout.flush()
        img = data[0].to(device)
        label = data[1].to(device)
        with torch.no_grad():
            pred = net(img)

        _, predicted = pred.max(1)
        y_pred.extend(predicted)
        y_true.extend(label)

    y_pred = torch.tensor(y_pred, device='cpu').numpy()
    y_true = torch.tensor(y_true, device='cpu').numpy()

    return y_pred, y_true


