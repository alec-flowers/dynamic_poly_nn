from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler, DataLoader


def load_db(name, transform, root='data', batch_size=64, shuffle=True, valid_ratio=0.2):
    trainset = getattr(datasets, name)(root=root, train=True, download=True, transform=transform)
    testset = getattr(datasets, name)(root=root, train=False, download=True, transform=transform)

    if valid_ratio > 0:
        # # divide the training set into validation and training set.
        instance_num = len(trainset)
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        valid_loader = None

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader
