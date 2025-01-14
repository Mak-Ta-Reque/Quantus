import random
random.seed(10)
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
import torchvision.transforms as T
transform = T.ToPILImage()
sys.path.insert(0, "/workspaces/Quantus/road_evaluation")
# Create a directory inside tmp for storing image
from road_evaluation.road import ImputedDataset, ThresholdDataset
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
        st.image(Image.open((image_data)),width=200)


uploaded_label = st.file_uploader("Choose label file", type='csv')
current_label_path = None
if uploaded_label is not None:
    #Save weight
    current_label_path = save_uploadedfile(uploadedfile=uploaded_label, path = label_path)


#Data set 



image_size_option = [(28, 28), (32, 32), (64, 64), (128, 128), (256, 256)]
IMAGE_SIZE = st.selectbox(
'Image size',
image_size_option, key="image_size")
print(type(IMAGE_SIZE))
transform_test = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #,
transdform_imputation = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
loader = config["DATA"]["loader"]
cifer_test = None
cifer_imputation = None
if not current_label_path is None:
    cifer_test = CSVdataset(csv_path=current_label_path, images_folder=image_dir, transform=transform_test)
    cifer_imputation = CSVdataset(csv_path=current_label_path, images_folder=image_dir, transform=transdform_imputation)

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





imputation_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
explantion = None
explantion_options = st.selectbox(
     'Which explantion do you want to use',
     (["ig", "ig_sg", "ig_sq", "ig_var"]))
if current_weight_path and next(iter(cifer_test)) and model:
    explantion, acc, outputs = generate_exp(cifer_test, 1, model, explantion_options)
    st.subheader(f"Model accuracy is {acc} % with output {outputs}")

ranking_option = st.selectbox(
        'Which ranking aproach',
        (["threshold", "sort"]))
if ranking_option == "threshold":
    threshold = True
elif ranking_option == "sort":
    threshold = False




exp_list = []
pert_method = None
if explantion:
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

    


    transform_test = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    accuracies_road_morf, probs = run_road(model, cifer_test, exp_list, transform_test, imputation_percentages, morf=True, imputation=pert_method, threshold=threshold)
    accuracies_road_lerf, probs = run_road(model, cifer_test, exp_list, transform_test, imputation_percentages, morf=False, imputation=pert_method, threshold=threshold)


    accuracies = {"MoRF": accuracies_road_morf,
                "LeRF": accuracies_road_lerf,
                "Amount removed":imputation_percentages  

    }
    st.line_chart(accuracies, x="Amount removed")
if cifer_imputation and pert_method :
    th_p = st.slider(
    'Select a threshold values/ for sort x/100 of the value',
    0.1, 0.9, 0.1)
    if threshold:
        ds_test_imputed_morf =ThresholdDataset(cifer_test,  mask=exp_list, th_p=th_p, remove=True, imputation = pert_method, transform=transform_test)
        ds_test_imputed_lerf =ThresholdDataset(cifer_test, mask=exp_list, th_p=th_p, remove=False, imputation = pert_method, transform=transform_test)
        ds_test_imputed_morf_h =ThresholdDataset(cifer_imputation,  mask=exp_list, th_p=th_p, remove=True, imputation = pert_method)
        ds_test_imputed_lerf_h =ThresholdDataset(cifer_imputation, mask=exp_list, th_p=th_p, remove=False, imputation = pert_method)
    
    else:
        ds_test_imputed_morf = ImputedDataset(cifer_test, mask=exp_list, th_p=th_p, remove=True, imputation = pert_method, transform=transform_test)
        ds_test_imputed_lerf = ImputedDataset(cifer_test,  mask=exp_list, th_p=th_p, remove=False, imputation = pert_method, transform=transform_test)
        ds_test_imputed_morf_h = ImputedDataset(cifer_imputation, mask=exp_list, th_p=th_p, remove=True, imputation = pert_method)
        ds_test_imputed_lerf_h = ImputedDataset(cifer_imputation,  mask=exp_list, th_p=th_p, remove=False, imputation = pert_method)
  
    model = model.eval()
    with torch.no_grad():
        output_morf = model(ds_test_imputed_morf[0][0].unsqueeze(dim=0))
        _, predicted_morf = torch.max(output_morf.data, 1)
        output_lerf = model(ds_test_imputed_lerf[0][0].unsqueeze(dim=0))
        _, predicted_lerf = torch.max(output_lerf.data, 1)
    print(np.linalg.norm(exp_list[0],axis=2).shape)
    st.image(transform(torch.tensor(np.linalg.norm(exp_list[0],axis=2))), width=200)
    st.image(transform(ds_test_imputed_morf_h[0][0]), width=200, caption=f"MoRF, T:{ds_test_imputed_morf[0][1]}, P: {predicted_morf}")
    st.image(transform(ds_test_imputed_lerf_h[0][0]), width=200, caption=f"LeRF, T:{ds_test_imputed_lerf[0][1]} P: {predicted_lerf}" )

# I am working here


