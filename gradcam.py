# TODO Dumb idea, technique works for CNN's, not for my type of neural network.

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import torch
import re
import glob
from utils import MODEL_PATH

from load_data import load_dataset
import nets

if __name__ == '__main__':

    paths = list(filter(
            lambda x: not re.search('xxxx', x),
            glob.glob(f"{MODEL_PATH}/CIFAR10/4/20220322-125017.ckpt", recursive=True)
        ))

    checkpoint = torch.load(paths[0])

    train_dataset, test_dataset = load_dataset(
            checkpoint['chosen_dataset'],
            checkpoint["transform"],
            apply_manipulation=checkpoint["apply_manipulation"],
            binary_class=checkpoint["binary_class"],
            ind_to_keep=checkpoint["ind_to_keep"],
        )

    print(f"Dataset: {type(test_dataset)}")
    sample_shape = test_dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(test_dataset.classes)
    channels_in = sample_shape[0]

    net = getattr(nets, checkpoint.get("net_name", "CCP"))(
        checkpoint["hidden_size"],
        image_size=image_size,
        n_classes=n_classes,
        channels_in=channels_in,
        n_degree=checkpoint["n_degree"]
    )
    net.load_state_dict(checkpoint['model_state_dict'])

    # net.eval()
    # sample = 0
    # input_tensor = test_dataset[sample][0] #.unsqueeze(0)
    # target_layers = [net.C]
    # targets = [ClassifierOutputTarget(test_dataset[sample][1])]
    # rgb_image = test_dataset[sample][0].permute((1, 2, 0)).squeeze()
    #
    # # output = net(input_tensor)
    # #
    # # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    # #
    # # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # # visualization = show_cam_on_image(_, grayscale_cam, use_rgb=True)
    def hook_store_A(module, input, output):
        module.A = output[0]
    def hook_store_dydA(module, grad_input, grad_output):
        module.dydA = grad_output[0]
    net.eval()

    layer = net.C
    layer.register_forward_hook(hook_store_A)
    layer.register_backward_hook(hook_store_dydA)

    sample = 0
    input_tensor = test_dataset[sample][0] #.unsqueeze(0)
    target = test_dataset[sample][1]

    output = net(input_tensor)
    output[0, target].backward()
    a = layer.dydA
    alpha = layer.dydA.mean((2, 3), keepdim = True)
    L = torch.relu((alpha * layer.A).sum(1, keepdim = True))
    L = L / L.max()
    L = torch.F.interpolate(L, size = (input.size(2), input.size(3)),
    mode = 'bilinear', align_corners = False)
    l = L.view(L.size(2), L.size(3)).detach().numpy()
