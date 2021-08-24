import streamlit as st

import my_functions as myf
from my_variables import *

import numpy as np
import random
from sklearn import preprocessing


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
  DISTANCES = st.multiselect('Distâncias', [1, 3, 5, 7, 9], [1, 3, 5])
  ANGLES = st.multiselect('Ângulos', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [0, np.pi/4, np.pi/2])
  
  st.markdown('### ORB')
  N_KEYPOINTS = st.slider('Número de keypoints', 1, 100, 10)

  st.markdown('### Gabor')
  KSIZE = st.slider('Ksize', 1, 100, 20)
  SIGMA = st.slider('Sigma', 1, 100, 20)
  THETA = st.slider('Theta', 0.01, 2*np.pi, 1*np.pi/4)
  LAMBDA = st.slider('Lambda', 0.01, 2*np.pi, 1*np.pi/4)
  GAMMA = st.slider('Gamma', 0.0, 1.0, 0.4, 0.05)
  PHI = st.slider('Phi', 0.0, 1.0, 0.0, 0.05)
  REDUCE_DIM = st.slider('Reduzir dimensões', 1, 32, 8)

  st.markdown('## PSO')
  N_PARTICLES = st.slider('Número de partículas', 1, 100, 15)
  ITERS = st.slider('Iterações', 1, 100, 5)

  submit_button = st.form_submit_button(label='Iniciar')

if submit_button:
  st.markdown('# Dataset')
  loading_dataset = st.empty()

  loading_dataset.markdown('Carregando dataset...')

  bar_dataset = st.progress(0)

  X, y_strings = myf.load_dataset(DATASET_FOLDER, progress_bar=bar_dataset)

  loading_dataset.markdown('Carregado.')
  bar_dataset.empty()

  le = preprocessing.LabelEncoder()
  le.fit(y_strings)
  y = le.transform(y_strings)

  resizing = st.empty()
  if X[0].shape[0] != SIZE:
    resizing.markdown(f'Redimensionando de {X[0].shape[0]} para {SIZE}...')
    f = lambda x: pre_process(x, SIZE)
    X = f(X)
    resizing.markdown(f'Redimensionado de {X[0].shape[0]} para {SIZE}.')

  st.markdown('## Exemplos')
  col1, col2, col3 = st.columns(3)

  with col1:
    n = random.randint(0, X.shape[0]-1)
    img = X[n]
    st.image(img)

  with col2:
    n = random.randint(0, X.shape[0]-1)
    img = X[n]
    st.image(img)

  with col3:
    n = random.randint(0, X.shape[0]-1)
    img = X[n]
    st.image(img)

  st.markdown('---')

  st.markdown(f'# Características - {FEATURE_EXTRACTOR}')

  loading_features = st.empty()
  loading_features.markdown(f'Extraindo características...')

  bar_features = st.progress(0)

  X_features = myf.feature_extractor(FEATURE_EXTRACTOR, X, progress_bar=bar_features)

  loading_features.markdown('Extraídas.')
  bar_features.empty()

  X_for_training = np.array(X_features)

  st.markdown('---')
  st.markdown(f'# Treino - {CLASSIFICATOR}')

  myf.main_fit(X_for_training, y)

  st.markdown('Finalizado.')