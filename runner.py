import sys
import torch
from torch.profiler import tensorboard_trace_handler

from plots import get_confusion_matrix


def train(net, train_loader, optimizer, criterion, epoch, device, writer, confusion_matrix, profiler=None):
    """ Perform single epoch of the training."""
    net.train()
    running_loss, correct, total, train_loss, acc = 0, 0, 0, 0, 0
    for idx, data_dict in enumerate(train_loader):
        img = data_dict[0]
        label = data_dict[1]
        inputs, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()
        optimizer.step()
        if profiler:
            profiler.step()

        total += label.size(0)
        running_loss += loss.item()

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(label).sum()

        train_loss = running_loss/total
        acc = float(correct)/total
        if idx % 100 == 0 and idx > 0:
            m2 = ('Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(epoch, idx, len(train_loader), float(train_loss), acc))

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", acc, epoch)
    # writer.flush()
    if confusion_matrix:
        writer.add_figure("Confusion/Train", get_confusion_matrix(train_loader, net, device), epoch)
        # writer.flush()


def train_profile(net, train_loader, optimizer, criterion, epoch, device, writer, confusion_matrix):
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
              profiler=profiler)


def test(net, test_loader, criterion, epoch, device, writer, confusion_matrix):
    """ Perform testing, i.e. run net on test_loader data
        and return the accuracy. """
    net.eval()
    correct, total, running_loss = 0, 0, 0
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
        correct += predicted.eq(label).sum().item()

    test_loss = running_loss / total
    acc = float(correct) / total

    writer.add_scalar("Loss/Test", test_loss, epoch)
    writer.add_scalar("Accuracy/Test", acc, epoch)
    # writer.flush()

    if confusion_matrix:
        writer.add_figure("Confusion/Test", get_confusion_matrix(test_loader, net, device), epoch)
        # writer.flush()

    print(f'Epoch {epoch} (Validation - Loss: {test_loss:.03f} & Accuracy: {acc:.03f}')


def load_checkpoint(net, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint Loaded - {checkpoint_path}")
