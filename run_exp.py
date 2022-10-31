import hydra
import os
import time
import quantus
from omegaconf import DictConfig, OmegaConf
import yaml
from yaml.loader import SafeLoader
import logging
import json
from collections import defaultdict
import torch

import torchvision.models as models
from collections.abc import Iterable
from typing import Union
from torchvision import transforms
import torchvision
from captum.attr import *
SMALL_OUTPUT = ['LayerGradCam']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
def get_model(**kwargs) -> torch.nn.Module:

    pretrained = kwargs.get("weight") #True if kwargs.get("weight") == "imagenet" else kwargs.get("weight")

    model_name = kwargs.get("model_name") if kwargs.get("model_name") is not None else "vgg16"
    print(kwargs)
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
    model.eval()
    return model
def get_layer(model= None, **kwargs):
    layer = eval("model.%s"%kwargs["layer"])

    return layer


def get_data(**kwargs):
    root = kwargs["root"]
    loader = kwargs["loader"]
    samples =  kwargs["n_samples"]
    size = eval(kwargs["size"])
    transformer = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    test_set = eval("torchvision.datasets.%s(root='%s', transform=transformer)"%(loader, root))
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=samples, pin_memory=True)
    x_batch, y_batch = iter(test_loader).next()
    return x_batch, y_batch



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
    #print(["%s : %s"%(i, saliency_outputs[i][1] ) for i in methods])


    # Faithfulness check
    matrics = config["config"]["EVALUATION"]
    matrics_obj ={}
    for matric in matrics:
        if matric == "RegionPerturbation":
            region_perturb = quantus.RegionPerturbation(**{
            "patch_size": 8,
            "regions_evaluation": 100,
            "img_size": image_size,
            "perturb_baseline": "uniform", })
            matrics_obj[matric] = region_perturb
        
        elif matric =="RegionPerturbationThreshold":
            region_perturb = quantus.RegionPerturbationThreshold(**{
            "patch_size": 0,
            "regions_evaluation": 20,
            "img_size": image_size,
            "perturb_baseline": "uniform", })
            matrics_obj[matric] = region_perturb

        
        else:
            raise NameError("Not implemented")
    x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

    run_time = {
    }
    """for key, val in  matrics_obj.items():
        results = {}
        for method in methods:
            time_1 = time.time()
        
            result = val(model=model, x_batch=x_batch,
                                    y_batch=y_batch,
                                    a_batch=saliency_outputs[method][0],
                                    **{"method":method, "device": device})
            time_2 = time.time()
            run_time[key] = [method, (time_2-time_1)]
            results[method] = result
        print(run_time)
        
        os.mkdir("%s/%s/"%(output_dir, key))
        val.plot(results=results, path_to_save= "%s/%s/%s.png"%(output_dir, key, key))
    """
    time_record = {}
    for key, metric_method in matrics_obj.items():
        results = {}
        time_consumption = {}
        for method in methods:
            time_1 = time.time()
            results[method] = metric_method(model = model, 
                                  x_batch = x_batch,
                                  y_batch = y_batch,
                                  a_batch = saliency_outputs[method][0],
                                  **{"explain_func": quantus.explain, "method": method, "device": device})
            time_2 = time.time()
            time_consumption[method] = (time_2 - time_1)
        time_record[key] = time_consumption
        metric_method.plot_gradient(results=results, path_to_save= "%s/AOPC_grad%s.png"%(output_dir, key))


    time_record = {"time": time_record}
    import json
    with open("%s/AOPC_time_%s.json"%(output_dir, key), 'w') as fp:
        json.dump(time_record, fp)

    
    """for key, metric_method in matrics_obj.items():
        results = {method: metric_method(model = model, 
                                  x_batch = x_batch,
                                  y_batch = y_batch,
                                  a_batch = saliency_outputs[method][0],
                                  **{"explain_func": quantus.explain, "method": method, "device": device}) for method in methods}
        metric_method.plot(results=results, path_to_save= "%s/AOPC_%s.png"%(output_dir, key))
    """


    
if __name__ == "__main__":
    main()