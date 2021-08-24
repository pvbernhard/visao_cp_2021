import streamlit as st
import my_functions as myf

import numpy as np
from sklearn import preprocessing


DATASET_FOLDER = 'Dataset'

feature_extractors = ['GLCM', 'ORB', 'Gabor']
classificators = ['XGBoost', 'Bagging']

FEATURE_EXTRACTOR = feature_extractors[0]
CLASSIFICATOR = classificators[0]

SIZE = 256

# GLCM

DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi/4, np.pi/2] # [0, np.pi/4] # np.pi/2, 3*np.pi/4, np.pi

# ORB

N_KEYPOINTS = 10

# Gabor

KSIZE = 20  #Use size that makes sense to the image and fetaure size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
SIGMA = 20 #Large sigma on small features will fully miss the features. 
THETA = 1*np.pi/4  #1/4 shows horizontal 3/4 shows other horizontal. Try other contributions
LAMBDA = 1*np.pi/4  #1/4 works best for angled. 
GAMMA = 0.4  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
PHI = 0  #Phase offset. I leave it to 0.
REDUCE_DIM = 8

# Cross-Validation

N_SPLITS = 5

# PSO

OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
N_PARTICLES = 15
DIMENSIONS = 5
X_MAX = 1 * np.ones(DIMENSIONS)
X_MAX[1:] *= 50
X_MIN = 0.01 * np.ones(DIMENSIONS)
BOUNDS = (X_MIN, X_MAX)
ITERS = 5

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