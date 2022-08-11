import quantus
from quantus.helpers import image_perturbation
import time
import hydra
from run_exp import get_data, get_model, get_layer
import numpy as np
from typing import Callable, Dict, List, Union
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
from captum.attr import *
import matplotlib.pyplot as plt
from quantus.helpers.perturb_func import baseline_replacement_by_mask

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> dict:
    config = yaml.load(OmegaConf.to_yaml(cfg),  Loader=SafeLoader)
    logging.info("\nModel: {MODEL}\nExplanation method: {EXPLANATION} \nData: {DATA}".format(**config["config"]))
    model_name = dict(**config["config"]["MODEL"])

    model = get_model(**model_name)
    explantion_layer = get_layer(model, **config["config"]["EXPLANATION"])
    x_batch, y_batch = get_data(**config["config"]["DATA"])
    image_size = eval(config["config"]["DATA"]["size"])
    methods = config["config"]["EXPLANATION"]["method"]
    output_dir = config["config"]["OUTPUT"]
    saliency_outputs = {} # Produced saliency maps
    #Produce faitfulness
    for method in methods:
        if method == "LayerGradCam":
            time_1 = time.time()
            a_batch_gradCAM = LayerGradCam(model, explantion_layer).attribute(inputs=x_batch, target=y_batch)
            a_batch = LayerAttribution.interpolate(a_batch_gradCAM, image_size).sum(axis=1).cpu().detach().numpy()
            a_batch = quantus.normalise_by_max(a_batch)    
            time_2 = time.time()
            saliency_outputs[method] = [a_batch, (time_2 - time_1)] # Saliency and the time to generate saliency

        elif method == "Saliency" :
            time_1 = time.time()
            a_batch = quantus.normalise_by_max(
            Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
            time_2 = time.time()
            saliency_outputs[method] = [a_batch, (time_2 - time_1)]

        elif method == "IntegratedGradients":
            time_1 = time.time()
            a_batch = quantus.normalise_by_max(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch,
            baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())
            time_2 = time.time()
            saliency_outputs[method] = [a_batch, (time_2 - time_1)]
        else:
            raise NameError("Not implemented")
    print(saliency_outputs["LayerGradCam"][0].shape)
    first_heatmap = saliency_outputs["LayerGradCam"][0]
    plt.imshow(first_heatmap[0], cmap='viridis')
    plt.colorbar()
  
    plt.savefig("%s/heatmap_CAM%s.png"%(output_dir, "1"), dpi=400)
    x_batch = x_batch.cpu().numpy()
    print(x_batch.shape)

    #pertureberd_image = image_perturbation.threshold_perturbation(x_batch=x_batch, a_batch=first_heatmap, level=20, sample_no=8)[0] #image_perturbation.region_perturbation(x_batch=x_batch, a_batch=first_heatmap, patch_size= 8, regions_evaluation=1000, order="morf", sample_no=999)[1]
    #pertureberd_image = image_perturbation.region_perturbation(x_batch=x_batch, a_batch=first_heatmap, patch_size= 8, regions_evaluation=1000, order="morf", sample_no=700)[0]
    pertureberd_image = image_perturbation.irof_perturbation(x_batch=x_batch, a_batch=first_heatmap, segmentation_method= "felzenszwalb", sample_no=1)[0]
    print(pertureberd_image.shape)
    pertureberd_image = np.moveaxis(pertureberd_image, 0, 2)
    plt.imshow(pertureberd_image, interpolation='nearest')
    plt.savefig("%s/perturbed%s.png"%(output_dir, "1"))
    print(np.linspace(0, 1, num=20))
    org_image = np.moveaxis(x_batch[0], 0, 2)
    plt.imshow(org_image, interpolation='nearest')
    plt.savefig("%s/perturbed%s.png"%(output_dir, "org_image"))

# generate segmantion map using threshol



# generate segmantation map using MORF
if __name__ == "__main__":
    main()

