import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch import mm

def norm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    else:
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return transform(img)

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = mm(features, features.t())
    return G.div(a * b * c * d)

def denorm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],std=[4.367, 4.464, 4.444])
        return transform(img)
    else:
        out = (img + 1) / 2
        return out.clamp(0, 1)
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
