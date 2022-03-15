import torch
from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset


def load_db(name, transform, root='data', batch_size=64, shuffle=True, valid_ratio=0.2, seed=0, subset=False):
    trainset = getattr(datasets, name)(root=root, train=True, download=True, transform=transform)
    testset = getattr(datasets, name)(root=root, train=False, download=True, transform=transform)
    g = torch.Generator()
    g.manual_seed(seed)

    if valid_ratio > 0:
        # # divide the training set into validation and training set.
        if subset:
            instance_num = 1000
        else:
            instance_num = len(trainset)
        print(f"Num Samples in train + validation: {instance_num}")
        indices = list(range(instance_num))
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, generator=g)
        valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, generator=g)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, generator=g)
        valid_loader = None

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader
