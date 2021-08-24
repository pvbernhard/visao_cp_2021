from my_variables import *


def pre_process(img, size):
  import cv2
  img = cv2.resize(img, (size, size))
  img = cv2.GaussianBlur(img, (5, 5), 0)
  return img


def feature_extractor_GLCM(dataset, progress_bar=None):
  import streamlit as st
  import pandas as pd
  from skimage.feature import greycomatrix, greycoprops

  st.markdown(f'Extraindo características...')
  feat_dataset = pd.DataFrame()
  
  n = 0
  for image_n in range(dataset.shape[0]):
    if progress_bar:
      progress_bar.progress(min((n / dataset.shape[0])*100, 99.9))
    
    df = pd.DataFrame()
    img = dataset[image_n, :, :]

    distances = DISTANCES
    angles = ANGLES

    i = 0
    for distance in distances:
      for angle in angles:
        GLCM = greycomatrix(img, [distance], [angle])       
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        df['Energy' + str(i)] = GLCM_Energy
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        df['Corr' + str(i)] = GLCM_corr       
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim' + str(i)] = GLCM_diss       
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        df['Homogen' + str(i)] = GLCM_hom       
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]
        df['Contrast' + str(i)] = GLCM_contr
        i += 1
    
    feat_dataset = feat_dataset.append(df)
    n += 1
  st.markdown(f'Shape: {feat_dataset.shape}')
  return feat_dataset


def feature_extractor_ORB(dataset, progress_bar=None):
  import streamlit as st
  import pandas as pd
  import cv2

  st.markdown('Extraindo características...')
  feat_dataset = pd.DataFrame()
  
  n = 0
  for image_n in range(dataset.shape[0]):
    if progress_bar:
      progress_bar.progress(min((n / dataset.shape[0])*100, 99.9))
    
    img = dataset[image_n, :, :]

    orb = cv2.ORB_create(N_KEYPOINTS)
    kp, des = orb.detectAndCompute(img, None)
    des = des.reshape(-1)
    df = pd.DataFrame(des)
    df = df.transpose()

    df /= 255.

    feat_dataset = feat_dataset.append(df)
    n += 1
  st.markdown(f'Shape: {feat_dataset.shape}')
  return feat_dataset


def feature_extractor_GABOR(dataset, progress_bar=None):
  import streamlit as st
  import pandas as pd
  import cv2

  st.markdown('Extraindo características...')
  feat_dataset = pd.DataFrame()

  ksize = KSIZE
  sigma = SIGMA
  theta = THETA
  lamda = LAMBDA
  gamma = GAMMA
  phi = PHI

  kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
  
  n = 0
  for image_n in range(dataset.shape[0]):
    if progress_bar:
      progress_bar.progress(min((n / dataset.shape[0])*100, 99.9))
    
    img = dataset[image_n, :, :]
    
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

    # reduzindo tamanho pra ter menos caracteristicas
    # pra reduzir o tempo
    fimg = cv2.resize(fimg, (SIZE // REDUCE_DIM, SIZE // REDUCE_DIM))

    fimg = fimg.reshape(-1)
    df = pd.DataFrame(fimg)
    df = df.transpose()

    df /= 255.

    feat_dataset = feat_dataset.append(df)
    n += 1
  st.markdown(f'Shape: {feat_dataset.shape}')
  return feat_dataset


def feature_extractor(feature_extractor, dataset, progress_bar=None):
  if feature_extractor == 'GLCM':
    return feature_extractor_GLCM(dataset, progress_bar)
  if feature_extractor == 'ORB':
    return  feature_extractor_ORB(dataset, progress_bar)
  if feature_extractor == 'Gabor':
    return  feature_extractor_GABOR(dataset, progress_bar)
  return None


def model_prediction(params, X_train, y_train, X_test):
  import xgboost as xgb

  model = xgb.XGBClassifier(
      use_label_encoder=False,
      eval_metric='logloss',
      eta=params[0], # range: [0,1]
      gamma=params[1], # range: [0,inf]
      max_depth=int(params[2]), # range:[+0,inf]
      min_child_weight=params[3], # range: [0,inf]
      max_delta_step=params[4], # range: [0,inf]
      )
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  return prediction


def model_prediction_bagging(params, X_train, y_train, X_test):
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import BaggingClassifier

  base_cls = DecisionTreeClassifier(max_depth=int(params[1]),
                                    min_samples_split=params[2],
                                    min_samples_leaf=params[3]
                                    )
  num_trees = int(params[4])
  model = BaggingClassifier(base_estimator = base_cls,
                            n_estimators = num_trees,
                            random_state = 42)
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  return prediction


def xgboost_cost(params, X_train, y_train, X_test, y_test):
  import numpy as np
  from sklearn import metrics

  results = []
  num_rows = np.shape(params)[0]

  params_to_use = params.copy()

  for i in range(num_rows):
    prediction = model_prediction(params_to_use[i], X_train, y_train, X_test)
    mean_error = metrics.mean_absolute_error(y_test, prediction)
    
    results.append(mean_error)

  return np.transpose(results)


def bagging_cost(params, X_train, y_train, X_test, y_test):
  import numpy as np
  from sklearn import metrics

  results = []
  num_rows = np.shape(params)[0]

  params_to_use = params.copy()

  for i in range(num_rows):
    prediction = model_prediction_bagging(params_to_use[i], X_train, y_train, X_test)
    mean_error = metrics.mean_absolute_error(y_test, prediction)
    
    results.append(mean_error)

  return np.transpose(results)


def load_dataset(dataset_folder, progress_bar=None):
  import os, glob, cv2

  X = []
  y = []

  total = 0
  for directory_path in glob.glob(os.path.join(dataset_folder, '*')):
    total += len(glob.glob(os.path.join(directory_path, "*.png")))

  counter = 0
  for directory_path in glob.glob(os.path.join(dataset_folder, '*')):
      label = directory_path.split(os.path.sep)[-1]
      for img_path in glob.glob(os.path.join(directory_path, "*.png")):
          img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
          img = pre_process(img, SIZE)
          X.append(img)
          y.append(label)
          counter += 1
          if progress_bar:
            progress_bar.progress(min((counter / (total + 1))*100, 99.9))

  X = np.array(X)
  y = np.array(y)
  return X, y


def main_fit(X, y):
  import streamlit as st
  from sklearn.model_selection import StratifiedKFold
  from pyswarms.single.global_best import GlobalBestPSO
  from sklearn import metrics

  skf = StratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)
  st.markdown(f'{skf}')

  accuracies = []
  mean_errors = []

  fold = 1
  for train_index, test_index in skf.split(X, y):

    st.markdown(f'Fold {fold}')

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.reshape(X_train.shape[0], -1)

    optimizer = GlobalBestPSO(n_particles=N_PARTICLES,
                              dimensions=DIMENSIONS,
                              options=OPTIONS,
                              bounds=BOUNDS)
    
    if CLASSIFICATOR == 'XGBoost':
      cost, pos = optimizer.optimize(xgboost_cost,
                                    iters=ITERS,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test
                                    )
      
      st.markdown(f'Parâmetros usados:')
      st.markdown(f'- max_depth={pos[1]}')
      st.markdown(f'- min_samples_split={pos[2]}')
      st.markdown(f'- min_samples_leaf={pos[3]}')
      st.markdown(f'- max_delta_step={pos[4]}')

      prediction = model_prediction(pos, X_train, y_train, X_test)

    elif CLASSIFICATOR == 'Bagging':
      cost, pos = optimizer.optimize(bagging_cost,
                                    iters=ITERS,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test
                                    )
      
      st.markdown(f'Parâmetros usados:')
      st.markdown(f'- eta={pos[0]}')
      st.markdown(f'- gamma={pos[1]}')
      st.markdown(f'- max_depth={pos[2]}')
      st.markdown(f'- min_child_weight={pos[3]}')
      st.markdown(f'- num_trees={pos[4]}')

      prediction = model_prediction_bagging(pos, X_train, y_train, X_test)
    else:
      raise Exception('Erro: classificador não encontrado.')

    acc = metrics.accuracy_score(y_test, prediction)
    log_loss = metrics.log_loss(y_test, prediction)
    mean_error = metrics.mean_absolute_error(y_test, prediction)
    st.markdown(f'Accuracy: {acc}')

    accuracies.append(acc)
    mean_errors.append(mean_error)

    st.markdown(metrics.classification_report(y_test, prediction, target_names=['benign', 'malignant']))

    st.markdown('---')

    fold += 1

  st.markdown(f'Média acc: {statistics.mean(accuracies)}')
  st.markdown(f'Média mean_error: {statistics.mean(mean_errors)}')
