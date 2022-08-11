import streamlit as st
import yaml
from yaml.loader import SafeLoader


st.title('Uber pickups in NNY')



# Open the file and load the file
with open('conf/config.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

config = data["config"]
data_options = config["DATA"]["root"]
option = st.selectbox(
     'Which dataset do you want to use',
     ('N/A', data_options))


st.write('You selected:', option)
