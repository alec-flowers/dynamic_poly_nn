import torch
from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset, Dataset

# Make it single class labels and see if we can improve accuracy...
# Look at papers for motivation behind this...


class OneClassDataset(Dataset):
    def __init__(self, dataset, binary_class, **kwargs):
        super(Dataset, self).__init__()
        self.binary_class = binary_class
        self.dataset = dataset
        self.classes = [self.dataset.classes[self.binary_class], "Other"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx][1] == self.binary_class:
            return self.dataset[idx][0], 0
        else:
            return self.dataset[idx][0], 1

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


class FewClassDataset(Dataset):
    def __init__(self, dataset, ind_to_keep, **kwargs):
        super(Dataset, self).__init__()
        self.ind_to_keep = ind_to_keep
        self.mapping_dict = {ind: i for i, ind in enumerate(self.ind_to_keep)}
        self.dataset = dataset
        self.classes = [*[self.dataset.classes[cls] for cls in self.ind_to_keep], "Other"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        old_label = self.dataset[idx][1]
        if old_label in self.mapping_dict:
            return self.dataset[idx][0], self.mapping_dict[old_label]
        else:
            return self.dataset[idx][0], len(self.ind_to_keep)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


def only_use_certain_class(dataset, ind_to_keep, **kwargs):

    # get indices for subset of classes to keep
    idx = torch.zeros(len(dataset), dtype=torch.bool)
    for i in ind_to_keep:
        tmp = (dataset.targets == i)
        idx = tmp | idx

    # overwrite target mapping to one starting from 0 and counting up to stay consistent with class_to_idx
    dataset.targets = dataset.targets[idx]
    mapping = {_class: i for i, _class in enumerate(ind_to_keep)}
    dataset.targets = dataset.targets.apply_({_class: i for i, _class in enumerate(ind_to_keep)}.get)

    # overwrite classes to keep only ones in our subset
    dataset.data = dataset.data[idx]
    dataset.classes = [dataset.classes[i] for i in range(len(dataset.classes)) if i in ind_to_keep]
    return dataset


def load_dataset(name, transform, root='data', apply_manipulation=None, **kwargs):
    trainset = getattr(datasets, name)(root=root, train=True, download=True, transform=transform)
    testset = getattr(datasets, name)(root=root, train=False, download=True, transform=transform)

    if apply_manipulation:
        trainset = apply_manipulation(trainset, **kwargs)
        testset = apply_manipulation(testset, **kwargs)

    return trainset, testset


# def split_and_load_dataloader(trainset, testset, batch_size=64, num_workers=0,shuffle=True, valid_ratio=0.2, seed=0, subset=False):
#     g = torch.Generator()
#     g.manual_seed(seed)
#
#
#
#     return train_loader, valid_loader, test_loader


def load_trainloader(trainset, batch_size=64, num_workers=0,shuffle=True, valid_ratio=0.2, seed=0, subset=False):
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

        # Note without persistent_workers=True, cost to restart new threads to do loading is brutal.
        train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, generator=g, num_workers=num_workers, persistent_workers=True)
        valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, generator=g, num_workers=num_workers, persistent_workers=True)
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers, persistent_workers=True)
        valid_loader = None

    return train_loader, valid_loader


def load_testloader(testset, batch_size=64, num_workers=0, shuffle=True, seed=0, persistent_workers=False):
    g = torch.Generator()
    g.manual_seed(seed)

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        persistent_workers= persistent_workers
    )
    return test_loader
