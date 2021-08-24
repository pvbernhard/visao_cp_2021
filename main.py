import streamlit as st

import my_functions as myf
from my_variables import *

import numpy as np
import random
from sklearn import preprocessing



def teste(args):
  print(args)

st.set_page_config(
  page_title='Trabalho de Visão CP',
  page_icon='💻',
  layout='wide',
  initial_sidebar_state='expanded'
)

with st.sidebar.form(key='form'):
  st.markdown('# Opções')
  FEATURE_EXTRACTOR = st.selectbox('Escolha o extrator de características', feature_extractors)
  CLASSIFICATOR = st.selectbox('Escolha o classificador', classificators)
  SIZE = st.number_input('Tamanho das imagens', 128, 460, 256)

  st.markdown('## Parâmetros')

  st.markdown('### GLCM')
  DISTANCES = st.multiselect('GLCM: Distâncias', [1, 3, 5, 7, 9], [1, 3, 5])
  ANGLES = st.multiselect('GLCM: Ângulos', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [0, np.pi/4, np.pi/2])
  
  st.markdown('### ORB')
  N_KEYPOINTS = st.slider('ORB: Número de keypoints', 1, 100, 10)

  st.markdown('### Gabor')
  KSIZE = st.slider('Gabor: Ksize', 1, 100, 20)
  SIGMA = st.slider('Gabor: Sigma', 1, 100, 20)
  THETA = st.slider('Gabor: Theta', 0.01, 2*np.pi, 1*np.pi/4)
  LAMBDA = st.slider('Gabor: Lambda', 0.01, 2*np.pi, 1*np.pi/4)
  GAMMA = st.slider('Gabor: Gamma', 0.0, 1.0, 0.4, 0.05)
  PHI = st.slider('Gabor: Phi', 0.0, 1.0, 0.0, 0.05)
  REDUCE_DIM = st.number_input('Gabor: Reduzir dimensões', 1, 32, 8)

  st.markdown('## PSO')
  N_PARTICLES = st.slider('Número de partículas', 1, 100, 15)
  ITERS = st.slider('Iterações', 1, 100, 5)

  submit_button = st.form_submit_button(label='Iniciar')

if submit_button:
  bar_dataset = st.progress(0)

  X, y_strings = myf.load_dataset(DATASET_FOLDER, progress_bar=bar_dataset)

  le = preprocessing.LabelEncoder()
  le.fit(y_strings)
  y = le.transform(y_strings)

  st.markdown('Exemplos:')
  for i in range(3):
    n = random.randint(0, X.shape[0]-1)
    img = X_base[n]
    st.image(img)

  bar_features = st.progress(0)

  X_features = myf.feature_extractor(FEATURE_EXTRACTOR, X, progress_bar=bar_features)

  X_for_training = np.array(X_features)

  myf.main_fit(X_for_training, y)