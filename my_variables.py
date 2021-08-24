import numpy as np

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