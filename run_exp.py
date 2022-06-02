import hydra
import quantus
from omegaconf import DictConfig, OmegaConf
import yaml
from yaml.loader import SafeLoader
import logging
import json
from collections import defaultdict
import torch
from os.path import exists
import torchvision.models as models
from collections.abc import Iterable
from typing import Union
from torchvision import transforms
import torchvision
SMALL_OUTPUT = ['LayerGradCam']
def get_model(**kwargs) -> torch.nn.Module:

    pretrained = kwargs.get("weight") #True if kwargs.get("weight") == "imagenet" else kwargs.get("weight")

    model_name = kwargs.get("name") if kwargs.get("name") is not None else "vgg16"
    try:
        if pretrained ==  "imagenet":
            logging.info("Loading model with imagenet weight: %s"%pretrained)
            model = eval("models.%s(pretrained=True)" %model_name)
        elif exists(pretrained):
            model = eval("models.%s()" % model_name)
            model.load_weight(pretrained)
        else:
            raise ValueError('Weight not a path or imagenet (%s)' % pretrained)

    except:
        # model = exec("models.%s()"%model_name)
        model = None
        raise ("Custom model Not implemented")
    logging.info("Model summery: \n%s"%model)
    return model
def get_layer(model= None, **kwargs):
    layer = eval("model.%s"%kwargs["layer"])

    return layer

def get_data(**kwargs):
    root = kwargs["root"]
    samples =  kwargs["n_samples"]
    transformer = transforms.Compose([transforms.ToTensor()])
    test_set = eval("torchvision.datasets.MNIST(root='%s', train=False, transform=transformer, download=True)"%root)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=samples, pin_memory=True)
    x_batch, y_batch = iter(test_loader).next()
    return x_batch, y_batch







@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> dict:
    config = yaml.load(OmegaConf.to_yaml(cfg),  Loader=SafeLoader)
    logging.info("\nModel: {MODEL}\nExplanation method: {EXPLANATION} \nData: {DATA}".format(**config["config"]))
    model_name = dict(**config["config"]["MODEL"])

    model = get_model(**model_name)
    explantion_layer = get_layer(model, **config["config"]["EXPLANATION"])
    data_x, data_y = get_data(**config["config"]["DATA"])

    methods = config["config"]["EXPLANATION"]["method"]
    print(methods)

    for method in methods:
        if method in SMALL_OUTPUT:
            a_batch_gradCAM = LayerGradCam(model, explantion_layer).attribute(inputs=data_x, target=data_x)
            a_batch = LayerAttribution.interpolate(a_batch_gradCAM, (28, 28)).sum(axis=1).cpu().detach().numpy()
        elif method == "Saliency" :
            a_batch_saliency = quantus.normalise_by_negative(
            Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
        elif method == "IntegratedGradients":
            quantus.normalise_by_negative(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch,
                                                                               baselines=torch.zeros_like(x_batch)).sum(
                axis=1).cpu().numpy())
        else:
            raise NameError("Not implemented")


if __name__ == "__main__":
    main()