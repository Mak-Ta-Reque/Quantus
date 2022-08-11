import streamlit as st

st.title('Uber pickups in NNY')

option = st.selectbox(
     'Which dataset do you want to use',
     ('Email', 'Home phone', 'Mobile phone'))


st.write('You selected:', option)