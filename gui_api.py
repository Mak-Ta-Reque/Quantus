import logging
import pandas as pd
import numpy as np
import torch
import logging
from os.path import exists
from torchvision import transforms
from torchvision import models as models
from torch import nn as nn
import torchvision
import time
from captum.attr import *
import quantus
import streamlit as st
from tqdm import tqdm
from road_evaluation.experiments.explanation_generation.utils import explanation_method
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
supported_models = ["vgg16", "resnet18"]
def output_perturbation(data_x, data_y, model, explantion, perturbation_method, saliency_method, amount):
    image_size = data_x[0].size()
    x_data, y_data = data_x.cpu().numpy(), data_y.cpu().numpy()
    if perturbation_method == "RegionPerturbation":
        region_perturb = quantus.RegionPerturbation(**{
        "patch_size": 8,
        "regions_evaluation": 1000,
        "img_size": image_size,
        "perturb_baseline": "uniform", })
        
    elif perturbation_method =="RegionPerturbationThreshold":
        region_perturb = quantus.RegionPerturbationThreshold(**{
        "patch_size": 0,
        "regions_evaluation":20,
        "img_size": image_size,
        "perturb_baseline": "uniform", })

    
    else:
        raise NameError("Not implemented")
    result = region_perturb(model = model, 
                                x_batch = x_data,
                                y_batch = y_data,
                                a_batch = explantion,
                                **{"explain_func": quantus.explain, "method": perturbation_method, "device": device})

    return result

def generate_exp(dataset, batch_size, model, explainer):
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        acc =  100 * correct / len(testloader.dataset)
        print('Accuracy of the network on test images: %.4f %%' % (
                    100 * correct / len(testloader.dataset)))

    cifar_train_expl = {}

    ## get explanation function
    get_expl = explanation_method(explainer)
    for i_num in tqdm(range(len(dataset)), position=0, leave=True):
        expl_dict = {}
        sample, clss = dataset[i_num]
        outputs = model(sample.unsqueeze(0).to(device))
        _, predicted = torch.max(outputs.data, 1)
        expl = get_expl(model, sample.unsqueeze(0).to(device), clss)
        # print(predicted.data[0].cpu().numpy(), outputs.data[0].cpu().numpy(), clss)
        expl_dict['expl'] = expl
        expl_dict['prediction'] = predicted.data[0].cpu().numpy()
        expl_dict['label'] = clss
        expl_dict['predict_p'] = outputs.data[0].cpu().numpy()
        cifar_train_expl[i_num] = expl_dict
    return cifar_train_expl, acc


@st.cache(allow_output_mutation=True)
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

@st.cache(allow_output_mutation=True)
def get_model_with_weight(model_name, weight, classes, device) -> torch.nn.Module:
    if not model_name in supported_models: raise Exception("The model is not supported")
    model = None
    if model_name == "resnet18":
        model = models.resnet18()
    if model_name == "vgg16":
        model = models.vgg16()
    
    try:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model = model.to(torch.device(device))
        # load trained classifier
        model.load_state_dict(torch.load(weight, map_location=torch.device(device)))
        model.eval()
    except Exception as e:
        logging.error("Model is not loaded or weight is not supported")
    return model

@st.cache(allow_output_mutation=True)
def get_layer(model= None, **kwargs):
    layer = eval("model.%s"%kwargs["layer"])
    return layer
    
@st.cache(allow_output_mutation=True)
def get_data(**kwargs):
    root = kwargs["data_dir"]
    loader = kwargs["loader"]
    samples =  kwargs["n_samples"]
    size = kwargs["size"]
    transformer = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    test_set = eval("torchvision.datasets.%s(root='%s', transform=transformer)"%(loader, root))
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=samples, pin_memory=True)
    return test_loader

@st.cache(allow_output_mutation=True)
def get_pertubation_output(**kwargs) -> pd.DataFrame:
    model = kwargs["model"]
    explantion_layer = kwargs["layer"]
    image_size =kwargs["data"][0][0][0].size()
    perurbation_method = kwargs["perturbation_option"]
    x_batch, y_batch = kwargs["data"]
    print(x_batch)
    results = {}
    #Produce faitfulness
    methods = kwargs["methods"]
    for method in methods:
        if method == "LayerGradCam":
            time_1 = time.time()
            a_batch_gradCAM = LayerGradCam(model, explantion_layer).attribute(inputs=x_batch, target=y_batch)
            print("gradcam: terminated before exp")
            a_batch = LayerAttribution.interpolate(a_batch_gradCAM, image_size, interpolate_mode="bilinear").sum(axis=1).cpu().detach().numpy()
            a_batch = quantus.normalise_by_max(a_batch)   
            print("terminated after exp") 
            time_2 = time.time()
            result = output_perturbation(x_batch, y_batch, model, a_batch, perurbation_method, method, 20)
            print("terminated after res")
            results[method] = [result, (time_2 - time_1)] # Saliency and the time to generate saliency
            print(time.time() - time_1)
        
        elif method == "Saliency" :
            time_1 = time.time()
            a_batch = quantus.normalise_by_max(
            Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
            time_2 = time.time()
            print("terminated after exp")
            result = output_perturbation(x_batch, y_batch, model, a_batch, perurbation_method, method,20)
            results[method] = [result, (time_2 - time_1)] # Saliency and the time to generate saliency

        elif method == "IntegratedGradients":
            time_1 = time.time()
            print("Integrated: terminated before exp")
            a_batch = quantus.normalise_by_max(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch,
                baselines=torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())
            time_2 = time.time()
            result = output_perturbation(x_batch, y_batch, model, a_batch, perurbation_method, method, 20)
            print("terminated after res")
            results[method] = [result, (time_2 - time_1)] # Saliency and the time to generate saliency
            print(time.time() - time_1)
        else:
            raise NameError("Not implemented")

    #print(["%s : %s"%(i, saliency_outputs[i][1] ) for i in methods])
    return results
  
    

def generate_aopc(**kwargs):
    model = kwargs["model"]
    explantion_layer = kwargs["layer"]
    dataloader = kwargs["data"]
    #image_size =kwargs["data"][0][0][0].size()
    perurbation_method = kwargs["perturbation_option"]
    n_batch = kwargs["n_smaples"]
    total_result = []
    for i, batch in enumerate(dataloader):
        #x_data, y_data = batch
        kwargs["data"] = batch
        batch_result = get_pertubation_output(**kwargs)
        
        results = {}
        for k, v in batch_result.items():
            #print(np.array(list(v.values())))
            results[k] = np.sum(np.array(list(v[0].values())), axis=0)/len(v[0].keys())
        total_result.append(results)
        
        if (i >= n_batch -1 ):
            print("All batch complete")
            break
    final_result = {k: np.zeros(len(v)) for k,v in total_result[0].items()}
    for res in total_result:
        for k, v in res.items():
            final_result[k] = final_result[k]  + v
    for k, v in final_result.items():
        final_result[k] = v/ n_batch
    
    grad_aopc = {}
    for k, v in final_result.items():
        grad_aopc[k] = np.gradient(v, kwargs["gradient_order"])

    return final_result, grad_aopc