from asyncio.log import logger
from cProfile import label
from crypt import methods
from email.policy import default
from functools import cache
from logging import root
from turtle import clone, mode
import numpy as np
import streamlit as st
import yaml
from yaml.loader import SafeLoader
import numpy as np
import pandas as pd
import gui_api
import copy
import logging
from captum.attr import *
import torch
import quantus
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
with open('conf/config.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

config = data["config"]
data_options = config["DATA"]["root"]

# Data selection
st.header("Give data information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    DATA_DIR = st.selectbox(
    'Data dierectory',
    (data_options, 'N/A'), key="data_dir")   

with col2:
    image_size_option = [(28, 28), (240, 240), (250, 250)]
    IMAGE_SIZE = st.selectbox(
    'Image size',
    image_size_option, key="image_size")

with col3:
    N_SAMPLES = st.number_input(
    label='Number of samples', min_value=1, key="n_samples")

with col4:
    N_BATCH = st.number_input(
    label='Number of batch', min_value=1, key="n_batch")
    st.session_state['batch'] = N_BATCH


loader = config["DATA"]["loader"]

try:
    data = gui_api.get_data(data_dir= DATA_DIR, loader=loader, n_samples=N_SAMPLES, size=IMAGE_SIZE)
    #logging.info(f"Number of samples: %d Shape of sample %s {len(data), str(x[0].size())}")
    st.session_state['data'] = iter(data)
except FileNotFoundError as e:
    logging.info(f"Error occured %s{e}")


st.header("Give model information")
col1, col2 = st.columns(2)

with col1:
    model_option = ["vgg16", "resnet50"]
    MODEL_NAME = st.selectbox(
    'Model name',
   model_option, key="model_name")   

with col2:
    weight_option = ["imagenet", "/path"]
    WEIGHT = st.selectbox(
    'Weight',
    weight_option, key="weight")

try:
    model = gui_api.get_model(model_name=MODEL_NAME, weight= WEIGHT)
except Exception as e:
    logging.info(f"Error occured %s{e}")

layer = None
if MODEL_NAME == "vgg16":
    layer =gui_api.get_layer(model = model, layer= "features[26]")

elif MODEL_NAME == "resnet50":
    layer =layer =gui_api.get_layer(model = model, layer= "layer4[2].conv3") 
st.session_state['layer'] = layer
logger.info(f"The name of explantion layer is {str(layer)}")

# Select all explantion mentod 
explantions = config["EXPLANATION"]["method"]
explantion_options = st.multiselect(
     'Which explantion do you want to use',
     (explantions))




perturbation_method = config["EVALUATION"]
perturbation_option = st.selectbox(
     'Which perturbation method do you want to use',
    perturbation_method)


# in this section we draw a perurbation curve

if st.button('Draw perturbation curve'):
    st.write('Perturbation curve')
    aopc_curve,  grad_aopc = gui_api.generate_aopc(model=model, layer=layer, data=data,
    methods=explantion_options, perturbation_option=perturbation_option,
    n_batch=N_BATCH, gradient_order=4)
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



if radio == 'ML model':
    st.write('ToDo.')
else:
    if st.button('next batch'):
        x_batch, y_batch = next(st.session_state['data'])
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
            print(a_batch)
        else:
            raise NameError("Not implemented")
        c = st.container()
        for image in a_batch:
            st.image(image, caption='Sunrise by the mountains', clamp=True)

        st.write("This is outside the container")
        
    



if st.button('Calculate'):
     st.write('Perturbation curve')
else:
     st.write('Click to measure the missclassification of user')