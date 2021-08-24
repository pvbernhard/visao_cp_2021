import streamlit as st

import my_functions as myf

import numpy as np
import random
from sklearn import preprocessing

from skimage.transform import resize


st.set_page_config(
  page_title='Trabalho de Visão CP',
  page_icon='💻',
  layout='wide',
  initial_sidebar_state='expanded'
)

with st.sidebar.form(key='form'):
  st.markdown('# Opções')
  myf.FEATURE_EXTRACTOR = st.selectbox('Escolha o extrator de características', feature_extractors)
  myf.CLASSIFICATOR = st.selectbox('Escolha o classificador', classificators)
  myf.SIZE = st.number_input('Tamanho das imagens', 32, 460, 256)

  st.markdown('## Parâmetros')

  st.markdown('### GLCM')
  myf.DISTANCES = st.multiselect('Distâncias', [1, 3, 5, 7, 9], [1, 3, 5])
  myf.ANGLES = st.multiselect('Ângulos', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [0, np.pi/4, np.pi/2])
  
  st.markdown('### ORB')
  myf.N_KEYPOINTS = st.slider('Número de keypoints', 1, 100, 10)

  st.markdown('### Gabor')
  myf.KSIZE = st.slider('Ksize', 1, 100, 20)
  myf.SIGMA = st.slider('Sigma', 1, 100, 20)
  myf.THETA = st.slider('Theta', 0.01, 2*np.pi, 1*np.pi/4)
  myf.LAMBDA = st.slider('Lambda', 0.01, 2*np.pi, 1*np.pi/4)
  myf.GAMMA = st.slider('Gamma', 0.0, 1.0, 0.4, 0.05)
  myf.PHI = st.slider('Phi', 0.0, 1.0, 0.0, 0.05)
  myf.REDUCE_DIM = st.slider('Reduzir dimensões', 1, 32, 8)

  st.markdown('## PSO')
  myf.N_PARTICLES = st.slider('Número de partículas', 1, 100, 15)
  myf.ITERS = st.slider('Iterações', 1, 100, 5)

  submit_button = st.form_submit_button(label='Iniciar')

if submit_button:
  st.markdown('# Dataset')
  loading_dataset = st.empty()

  loading_dataset.markdown('Carregando dataset...')

  bar_dataset = st.progress(0)

  X, y_strings = myf.load_dataset(myf.DATASET_FOLDER, myf.SIZE, progress_bar=bar_dataset)

  loading_dataset.markdown('Carregado.')
  bar_dataset.empty()

  st.markdown(f'Shape do dataset: {X.shape}')

  le = preprocessing.LabelEncoder()
  le.fit(y_strings)
  y = le.transform(y_strings)

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

  st.markdown(f'# Características - {myf.FEATURE_EXTRACTOR}')

  loading_features = st.empty()
  loading_features.markdown(f'Extraindo características...')

  bar_features = st.progress(0)

  X_features = myf.feature_extractor(myf.FEATURE_EXTRACTOR, X, progress_bar=bar_features)

  loading_features.markdown('Extraídas.')
  bar_features.empty()

  st.markdown(f'Shape das características: {X_features.shape}')

  X_for_training = np.array(X_features)

  st.markdown('---')
  st.markdown(f'# Treino - {myf.CLASSIFICATOR}')

  myf.main_fit(X_for_training, y)

  st.markdown('Finalizado.')