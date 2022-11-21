from asyncio.log import logger
import torchvision.transforms as transforms
from email.policy import default
from logging import root
from turtle import clone, mode
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import gui_api
import copy
import logging
from captum.attr import *
import torch
import quantus
from quantus.helpers import image_perturbation
import os
from dataset import CSVdataset
from gui_api import generate_exp

import sys
sys.path.insert(0, "/workspaces/Quantus/road_evaluation")
# Create a directory inside tmp for storing image

def del_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

image_dir = f"/tmp/images"
weight_path = "/tmp/weight"
label_path = "/tmp/label"


if not os.path.exists(image_dir):
    os.mkdir(image_dir)

if not os.path.exists(weight_path):
    os.mkdir(weight_path)

if not os.path.exists(label_path):
    os.mkdir(label_path)

def load_image(image_file):
	img = Image.open(image_file)
	return img

del_files(image_dir)
del_files(weight_path)
del_files(label_path)

def save_uploadedfile(uploadedfile, path= image_dir):
    file_path = os.path.join(path, uploadedfile.name)
    with open(file_path,"wb") as f:
         f.write(uploadedfile.getbuffer())
    return file_path
st.title('Heatmap evaluation tool')
IMAGE_SIZE = (240, 240)
DATA_DIR = None
N_SAMPLES = 1
MODEL_NAME = None
WEIGHT = None
N_BATCH = 1
if 'perturbations' not in st.session_state:
    st.session_state['perturbations'] = 1

if 'batch' not in st.session_state:
    st.session_state['batch'] = 2

if "data" not in st.session_state:
    st.session_state['data'] = None

if "layer" not in st.session_state:
    st.session_state['layer'] = None


# Open the file and load the file
with open('conf/config_webapp.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

config = data["config"]
data_options = config["DATA"]["root"]


# Data selection
st.header("Give data information")
uploaded_files = st.file_uploader("Choose a jpg/png file", accept_multiple_files=True, type=["jpg", "png"])
for image_data in uploaded_files:
    if image_data is not None:
        file_details = f"Image name: {image_data.name}, size: {image_data.size}"
        #st.image(load_image(image_data),width=250)
        save_uploadedfile(uploadedfile=image_data, path=image_dir)

uploaded_label = st.file_uploader("Choose label file", type='csv')
current_label_path = None
if uploaded_label is not None:
    #Save weight
    current_label_path = save_uploadedfile(uploadedfile=uploaded_label, path = label_path)


#Data set 
transform_test = transforms.Compose([transforms.Resize(35),
   transforms.CenterCrop(32), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
if not current_label_path is None:
    cifer_test = CSVdataset(csv_path=current_label_path, images_folder=image_dir, transform=transform_test)


image_size_option = [(28, 28), (64, 64), (128, 128), (256, 256)]
IMAGE_SIZE = st.selectbox(
'Image size',
image_size_option, key="image_size")

loader = config["DATA"]["loader"]

try:
    data = gui_api.get_data(data_dir= DATA_DIR, loader=loader, n_samples=1, size=IMAGE_SIZE)
    st.session_state['data'] = data
    
except FileNotFoundError as e:
    logging.info(f"Error occured %s{e}")


st.header("Give model information")
col1, col2, col3 = st.columns(3)

with col1:
    MODEL_NAME = st.selectbox(
    'Model name',
  gui_api.supported_models, key="model_name")   
with col3:
    classes = st.number_input("Number of classes", 1, key="classes")

with col2:
    device = st.selectbox(
    'Device',
    ["cpu", "gpu0"], key="device")

uploaded_weight = st.file_uploader("Choose .torch weight file", type=['pth', 'toch'])
current_weight_path = None
if uploaded_weight is not None:
    #Save weight
    current_weight_path = save_uploadedfile(uploadedfile=uploaded_weight, path = weight_path)

model = gui_api.get_model_with_weight(MODEL_NAME, current_weight_path, classes, device)
if model is None:
    st.warning("Selct the model and weight appropiately")

layer = None
if MODEL_NAME == "vgg16":
    layer =gui_api.get_layer(model = model, layer= "features[26]")

elif MODEL_NAME == "resnet50":
    layer =layer =gui_api.get_layer(model = model, layer= "layer4[2].conv3") 
st.session_state['layer'] = layer
logger.info(f"The name of explantion layer is {str(layer)}")



# Select all explantion mentod 


perturbation_option = st.selectbox(
     'Which perturbation method do you want to use',
    ["Noisy linear (SOTA)", "Telea", "NS", "Zero"])





imputation_percentages = [0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995]

explantion_options = st.selectbox(
     'Which explantion do you want to use',
     (["ig", "ig_sg", "ig_var"]))
if current_weight_path and next(iter(cifer_test)) and model:
    explantion, acc = generate_exp(cifer_test, 1, model, explantion_options)
    st.subheader(f"Model accuracy is {acc} %")

exp_list = []
for key, val in explantion.items():
    exp_list.append(val["expl"])
from  road_evaluation.road.imputations import NoisyLinearImputer, ZeroImputer, GAINImputer, ChannelMeanImputer, ImpaintingImputation, ImpaintingImputationNS
from road_evaluation.road import run_road

if perturbation_option == "Noisy linear (SOTA)":
    pert_method = NoisyLinearImputer()
elif perturbation_option == "Zero":
    pert_method = ZeroImputer()
elif perturbation_option == "Telea":
    pert_method = ImpaintingImputation()
elif perturbation_option == "NS":
    pert_method = ImpaintingImputationNS()
else:
    raise NotImplementedError(f"{perturbation_option} is not implemented")
# update transform_test for adapting with the road

ranking_option = st.selectbox(
     'Which ranking aproach',
     (["threshold", "sort"]))
if ranking_option == "threshold":
    threshold = True
elif ranking_option == "sort":
    threshold = False


transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
accuracies_road_morf, probs = run_road(model, cifer_test, exp_list, transform_test, imputation_percentages, morf=True, imputation=pert_method, threshold=threshold)
accuracies_road_lerf, probs = run_road(model, cifer_test, exp_list, transform_test, imputation_percentages, morf=False, imputation=pert_method, threshold=threshold)
if not threshold:
   accuracies_road_lerf =  torch.flip(accuracies_road_lerf, dims =[0])

accuracies = {"MoRF": accuracies_road_morf,
            "LeRF": accuracies_road_lerf

}
st.line_chart(accuracies)

# I am working here


# in this section we draw a perurbation curve
if st.button('Draw perturbation curve'):
    st.write('Perturbation curve')
    x_train = []
    y_train = []
    data = iter(data)
    for i in range(N_SAMPLES):
        x, y = data.next()
        x_train += x
        y_train += y 
    
    aopc_curve,  grad_aopc = gui_api.generate_aopc(model=model, layer=layer, data=st.session_state['data'],
        methods=explantion_options, perturbation_option=perturbation_option,
        gradient_order=4, n_smaples = N_SAMPLES)
    st.line_chart(aopc_curve)
    st.line_chart(grad_aopc)
    st.write('Perturbation curve amd its gradient')
    st.session_state['perturbations'] = len(list(aopc_curve.values())[0])

else:
     st.write('Click the above to generate perturbation curve')

st.header("User study part")

saliency_method = config["EXPLANATION"]["method"]
option = st.selectbox(
     'Which saliency method do you want to use',
    saliency_method)

st.write('You selected: ', option)


perturbation_method = config["EVALUATION"]
perturbation_option = st.selectbox(
     'Which perturbation method do you want to use',
    perturbation_method, key="user_study_pertubation_option")


threshold = st.slider('Amount of perturbation (When the gradient of perturbation curve is maximum)?', 0, 1, st.session_state['perturbations'] )
st.write("Perturbation at  ", threshold)
image_list = range(20)
radio =  st.radio(
        "Evaluator",
        ('ML model', 'Human'))

print(data)

if radio == 'ML model':
    st.write('ToDo.')
else:
    data = iter(st.session_state['data'])
    if st.button('Feature removal'):

        for i in range(N_SAMPLES):
            x_batch, y_batch = next(data)
            a_batch = None
            
            if option == "LayerGradCam":
                a_batch_gradCAM = LayerGradCam(model, st.session_state['layer']).attribute(inputs=x_batch, target=y_batch)
                a_batch = LayerAttribution.interpolate(a_batch_gradCAM, IMAGE_SIZE, interpolate_mode="bilinear").sum(axis=1).cpu().detach().numpy()
                a_batch = quantus.normalise_by_max(a_batch)
                
                
            elif option == "Saliency" :
                a_batch = quantus.normalise_by_max(
                    Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
                
            elif option == "IntegratedGradients":
                a_batch = quantus.normalise_by_max(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch,
                    baselines = torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())
               
            else:
                raise NameError("Not implemented")

            x_batch = x_batch.cpu().numpy()
            if perturbation_option == "RegionPerturbationThreshold":
                distorted_image = image_perturbation.threshold_perturbation(x_batch=x_batch, a_batch=a_batch, level=20, sample_no=threshold)
            elif perturbation_option == "RegionPerturbation":
                distorted_image = image_perturbation.region_perturbation(x_batch=x_batch, a_batch=a_batch,patch_size= 8, regions_evaluation=700, order="morf")
            else:
                st.write("not implemented")
            c = st.container()
            distorted_image =np.moveaxis( distorted_image[0], 0,2) #np.moveaxis(distorted_image[0],0, 2)
            print(distorted_image.shape)
            st.image(distorted_image, caption='Important feture removed', clamp=True)
            


        st.write("This is outside the container")




if st.button('Original'):
    data = iter(st.session_state['data'])

    for i in range(N_SAMPLES):
        x_batch, y_batch = next(data)
        a_batch = None
        
        if option == "LayerGradCam":
            a_batch_gradCAM = LayerGradCam(model, st.session_state['layer']).attribute(inputs=x_batch, target=y_batch)
            a_batch = LayerAttribution.interpolate(a_batch_gradCAM, IMAGE_SIZE).sum(axis=1).cpu().detach().numpy()
            a_batch = quantus.normalise_by_max(a_batch)
            
            
        elif option == "Saliency" :
            a_batch = quantus.normalise_by_max(
                Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1).cpu().numpy())
            
        elif option == "IntegratedGradients":
            a_batch = quantus.normalise_by_max(IntegratedGradients(model).attribute(inputs=x_batch, target=y_batch,
                baselines = torch.zeros_like(x_batch)).sum(axis=1).cpu().numpy())
        
        else:
            raise NameError("Not implemented")

        x_batch = x_batch.cpu().numpy()

        distorted_image = image_perturbation.threshold_perturbation(x_batch=x_batch, a_batch=a_batch, level=20, sample_no=threshold)
        c = st.container()
        distorted_image =np.moveaxis( x_batch[0], 0,2) #np.moveaxis(distorted_image[0],0, 2)
        print(distorted_image.shape)
        st.image(distorted_image, caption='Important feture removed', clamp=True)
            

else:
     st.write('Click to measure the missclassification of user')
