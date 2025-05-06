# *-* encoding=utf8 *-*

# El propósito del laboratorio es familiarizarse con el algoritmo de «Neural
# Style» o «Neural Transfer» que permite reproducir una imagen en un estilo
# artístico diferente. Para ello toma:
#
#  * Imagen de entrada
#  * Imagen de contenido
#  * Imagen de estilo
#
# Y lo que hace es modificar la de la entrada para que sea como la de contenido
# en el estilo de la de estilo.

import torch
import torch.nn
import torch.nn.functional
import torch.optim

import PIL
import matplotlib.pyplot

import torchvision.transforms
import torchvision.models

import copy
import os
import numpy


# ==== Metaprogramming ====================================================== #

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


# ==== Precomputed global values ============================================ #

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
    """ This function returns the cachéd pretrained model """
    return val

@Closure(val=torch.tensor([0.485, 0.456, 0.406]).to(device()))
def cnn_normalization_mean(val):
    return val

@Closure(val=torch.tensor([0.229, 0.224, 0.225]).to(device()))
def cnn_normalization_std(val):
    return val


# ==== Image ================================================================ #

@ClassClosure(
    loader = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((image_size(), image_size())),
         torchvision.transforms.ToTensor()]),
    unloader = torchvision.transforms.ToPILImage())
class Image:
    """ """
    def __init__ (self, image : object = None, path : str = None):
        # La imagen de PIL tiene formato `Unsigned_8'. Torch por debajo utiliza
        # un tipo racional en el rango `0.0 .. 1.0'.
        if path is None and image is None:
            raise Exception ("path or image shoud be something >:(")
        elif path is not None and image is not None:
            raise Exception ("path or image not both >:(")
        if path:
            image = self.loader(PIL.Image.open(path)).unsqueeze(0)
            self.__image = image.to(device(), torch.float)
        else:
            self.__image = image

    @property
    def image (self):
        return self.unloader(self.__image.cpu().clone().squeeze(0))

    @property
    def raw_image (self):
        return self.__image

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

    def clone (self):
        return Image(image=self.__image.clone())
        

# ==== Classes I don't understand and I don't pretend to ==================== #

class ContentLoss(torch.nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=[f"conv_{i}" for i in range(1, 6)]):
    # NOTE: Should I remove this shit
    cnn = copy.deepcopy(cnn())

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device())

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = torch.nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img.raw_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,print_step=50,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % print_step == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return Image(image=input_img)
