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

form_vars = {
  # default values

  'DATASET_FOLDER': 'Dataset',

  'feature_extractors': ['GLCM', 'ORB', 'Gabor'],
  'classificators': ['XGBoost', 'Bagging'],


  'SIZE': 256,

  # GLCM

  'DISTANCES': [1, 3, 5],
  'ANGLES': [0, np.pi/4, np.pi/2], # [0, np.pi/4] # np.pi/2, 3*np.pi/4, np.pi

  # ORB

  'N_KEYPOINTS': 10,

  # Gabor

  'KSIZE': 20,  #Use size that makes sense to the image and fetaure size. Large may not be good. 
  #On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
  'SIGMA': 20,  #Large sigma on small features will fully miss the features. 
  'THETA': 1*np.pi/4,  #1/4 shows horizontal 3/4 shows other horizontal. Try other contributions
  'LAMBDA': 1*np.pi/4,  #1/4 works best for angled. 
  'GAMMA': 0.4,  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
  #Value of 1, spherical may not be ideal as it picks up features from other regions.
  'PHI': 0,  #Phase offset. I leave it to 0.
  'REDUCE_DIM': 8,

  # Cross-Validation

  'N_SPLITS': 5,

  # PSO

  'OPTIONS': {
    'c1': 0.5,
    'c2': 0.3,
    'w':0.9
  },
  'N_PARTICLES': 15,
  'DIMENSIONS': 5,
  'X_MAX': 1 * np.ones(DIMENSIONS),
  'X_MIN': 0.01 * np.ones(DIMENSIONS),
  'ITERS': 5
}

form_vars['FEATURE_EXTRACTOR'] = form_vars.get('feature_extractors')[0]
form_vars['CLASSIFICATOR'] = form_vars.get('classificators')[0]
form_vars['X_MAX'][1:] *= 50
form_vars['BOUNDS'] = (form_vars.get('X_MIN'), form_vars.get('X_MAX'))

with st.sidebar.form(key='form'):
  st.markdown('# Opções')
  form_vars['FEATURE_EXTRACTOR'] = st.selectbox('Escolha o extrator de características', feature_extractors)
  form_vars['CLASSIFICATOR'] = st.selectbox('Escolha o classificador', classificators)
  form_vars['SIZE'] = st.number_input('Tamanho das imagens', 32, 460, 256)

  st.markdown('## Parâmetros')

  st.markdown('### GLCM')
  form_vars['DISTANCES'] = st.multiselect('Distâncias', [1, 3, 5, 7, 9], [1, 3, 5])
  form_vars['ANGLES'] = st.multiselect('Ângulos', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4], [0, np.pi/4, np.pi/2])
  
  st.markdown('### ORB')
  form_vars['N_KEYPOINTS'] = st.slider('Número de keypoints', 1, 100, 10)

  st.markdown('### Gabor')
  form_vars['KSIZE'] = st.slider('Ksize', 1, 100, 20)
  form_vars['SIGMA'] = st.slider('Sigma', 1, 100, 20)
  form_vars['THETA'] = st.slider('Theta', 0.01, 2*np.pi, 1*np.pi/4)
  form_vars['LAMBDA'] = st.slider('Lambda', 0.01, 2*np.pi, 1*np.pi/4)
  form_vars['GAMMA'] = st.slider('Gamma', 0.0, 1.0, 0.4, 0.05)
  form_vars['PHI'] = st.slider('Phi', 0.0, 1.0, 0.0, 0.05)
  form_vars['REDUCE_DIM'] = st.slider('Reduzir dimensões', 1, 32, 8)

  st.markdown('## PSO')
  form_vars['N_PARTICLES'] = st.slider('Número de partículas', 1, 100, 15)
  form_vars['ITERS'] = st.slider('Iterações', 1, 100, 5)

  submit_button = st.form_submit_button(label='Iniciar')

if submit_button:
  st.markdown('# Dataset')
  loading_dataset = st.empty()

  loading_dataset.markdown('Carregando dataset...')

  bar_dataset = st.progress(0)

  X, y_strings = myf.load_dataset(form_vars.get('DATASET_FOLDER'), form_vars.get('SIZE'), progress_bar=bar_dataset)

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

  st.markdown(f"# Características - {form_vars.get('FEATURE_EXTRACTOR')}")

  loading_features = st.empty()
  loading_features.markdown(f'Extraindo características...')

  bar_features = st.progress(0)

  X_features = myf.feature_extractor(form_vars, X, progress_bar=bar_features)

  loading_features.markdown('Extraídas.')
  bar_features.empty()

  st.markdown(f'Shape das características: {X_features.shape}')

  X_for_training = np.array(X_features)

  st.markdown('---')
  st.markdown(f"# Treino - {form_vars.get('CLASSIFICATOR')}")

  myf.main_fit(form_vars, X_for_training, y)

  st.markdown('Finalizado.')