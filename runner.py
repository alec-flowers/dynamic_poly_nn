import sys
import torch


def train(net, train_loader, optimizer, criterion, epoch, device, writer):
    """ Perform single epoch of the training."""
    net.train()
    running_loss, correct, total = 0, 0, 0
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

        total += label.size(0)
        running_loss += loss.item()

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(label).cpu().sum()

        train_loss = running_loss/total
        acc = float(correct)/total
        if idx % 100 == 0 and idx > 0:
            m2 = ('Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(epoch, idx, len(train_loader), float(train_loss), acc))

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", acc, epoch)
        writer.flush()
    return running_loss


def test(net, test_loader, criterion, epoch, device, writer):
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
    writer.flush()

    print(f'Epoch {epoch} (Validation - Loss: {test_loss:.03f} & Accuracy: {acc:.03f}')