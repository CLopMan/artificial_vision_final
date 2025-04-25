# *-* encoding=utf8 *-*

import torch
import torch.nn
import torch.nn.functional
import torch.optim

import PIL
import matplotlib.pyplot

import torchvision.transforms as transforms
import torchvision.models

import copy
import os
import numpy as np


class Closure:
    """
    Function decorator that puts values into a closure in order to avoid
    recomputations. For instance:

    @Closure(x=computationally_complex_func(42), y=10):
    def my_func(a, b, c, x, y):
        return a * x + b * y - c
    """
    def __init__ (self, **kwargs):
        self.__kwargs = kwargs
    def __call__ (self, func):
        def inner (*args, **kwargs):
            return func(*args, **kwargs, **self.__kwargs)
        return inner

class ClassClosure:
    """
    Class decorator equivalent to the `Closure' class. This one stores the
    given arguments a public elements of the given class.

    @ClassClosure(foo=10, bar=func(42)):
    class MyClass:
        def __init__(self):
            print(self.foo)
            print(self.bar)
    """
    def __init__ (self, **kwargs):
        self.__kwargs = kwargs
    def __call__ (self, cls):
        params = self.__kwargs
        class Inner(cls):
            def __init__(self, *args, **kwargs):
                self.__dict__.update(params)
                super().__init__(*args, **kwargs)
        return Inner

@Closure(dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
def device (dev : object) -> object:
    """ This function returns the torch device """
    return dev

@Closure(size=512 if torch.cuda.is_available() else 128)
def image_size (size : int) -> int:
    """ This function sugests an appropiate image size """
    return size

@Closure(val=torchvision.models.vgg19(pretrained=True).features.to(device()).eval())
def cnn(val):
    """ This function returns the cachéd neural network """
    return val

@ClassClosure(
    loader = transforms.Compose([transforms.Resize((image_size(), image_size())),
                                 transforms.ToTensor()]),
    unloader = transforms.ToPILImage())
class Image:
    """ """
    def __init__ (self, path : str):
        img = self.loader(PIL.Image.open(path)).unsqueeze(0)
        self.__image = img.to(device(), torch.float)

    @property
    def image (self):
        return self.unloader(self.__image.cpu().clone().squeeze(0))

    def show (self, title="", ax=None):
        image = self.image
        if ax is None:
            matplotlib.pyplot.imshow(image)
            matplotlib.title(title)
        else:
            ax.imshow(image)
            ax.set_title(title)

    def save (self, path : str):
        self.image.save(path)


class ContentLoss(torch.nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(torch.nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

